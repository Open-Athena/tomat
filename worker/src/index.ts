/**
 * tomat-runs-api — CFW backing tomat.oa.dev/runs.
 *
 * Endpoints (all read-only, public for now):
 *   GET  /api/runs                       — list of synced run ids
 *   GET  /api/runs/:id/manifest.json     — per-run metadata (config, summary, history range)
 *   GET  /api/runs/:id/raw.parquet       — full history parquet
 *   GET  /health
 *
 * Backed by R2 `openathena/tomat/runs/<id>/{raw.parquet,manifest.json}`,
 * populated out-of-band by `tomat runs sync <substr>` (will become an
 * on-demand pull from wandb in a later phase — see specs/23-runs-dashboard.md).
 */

export interface Env {
	R2: R2Bucket;
	CORS_ORIGIN: string;
	R2_RUNS_PREFIX: string;
}

function corsHeaders(env: Env): HeadersInit {
	return {
		'Access-Control-Allow-Origin': env.CORS_ORIGIN,
		'Access-Control-Allow-Methods': 'GET, OPTIONS',
		'Access-Control-Max-Age': '86400',
	};
}

function jsonResponse(data: unknown, env: Env, init?: ResponseInit): Response {
	return new Response(JSON.stringify(data), {
		...init,
		headers: {
			'Content-Type': 'application/json',
			'Cache-Control': 'public, max-age=60',
			...corsHeaders(env),
			...(init?.headers ?? {}),
		},
	});
}

async function serveR2Object(req: Request, env: Env, key: string): Promise<Response> {
	// Honor Range requests — required for hyparquet, which fetches the
	// parquet footer first before issuing typed-column reads.
	const rangeHeader = req.headers.get('Range');
	const r2Range = parseRangeHeader(rangeHeader);
	const obj = r2Range
		? await env.R2.get(key, { range: r2Range })
		: await env.R2.get(key);
	if (!obj) {
		return new Response(`Not found: ${key}`, {
			status: 404,
			headers: corsHeaders(env),
		});
	}
	const headers = new Headers();
	obj.writeHttpMetadata(headers);
	headers.set('etag', obj.httpEtag);
	headers.set('Cache-Control', 'public, max-age=60');
	headers.set('Accept-Ranges', 'bytes');
	const totalSize = obj.size;
	let status = 200;
	if (r2Range) {
		// R2 returned a partial body; compute the actual byte range.
		let start: number, end: number;
		if ('suffix' in r2Range && typeof r2Range.suffix === 'number') {
			start = Math.max(0, totalSize - r2Range.suffix);
			end = totalSize - 1;
		} else {
			const offsetRange = r2Range as { offset?: number; length?: number };
			start = offsetRange.offset ?? 0;
			end = start + (offsetRange.length ?? totalSize - start) - 1;
		}
		headers.set('Content-Range', `bytes ${start}-${end}/${totalSize}`);
		headers.set('Content-Length', `${end - start + 1}`);
		status = 206;
	} else {
		headers.set('Content-Length', `${totalSize}`);
	}
	for (const [k, v] of Object.entries(corsHeaders(env))) {
		headers.set(k, v as string);
	}
	// For HEAD requests, the runtime drops the body automatically.
	return new Response(obj.body, { status, headers });
}

/** Parse an HTTP Range header into the R2.get options shape. Supports
 * `bytes=START-END`, `bytes=START-`, `bytes=-SUFFIX`. Returns undefined
 * if no header or unparseable (caller falls back to full-object read). */
function parseRangeHeader(h: string | null): R2Range | undefined {
	if (!h) return undefined;
	const m = h.match(/^bytes=(\d*)-(\d*)$/);
	if (!m) return undefined;
	const [, startS, endS] = m;
	if (startS === '' && endS !== '') {
		// suffix range: bytes=-N (last N bytes)
		return { suffix: parseInt(endS, 10) };
	}
	if (startS !== '' && endS === '') {
		// open-ended: bytes=START-
		return { offset: parseInt(startS, 10) };
	}
	if (startS !== '' && endS !== '') {
		const start = parseInt(startS, 10);
		const end = parseInt(endS, 10);
		return { offset: start, length: end - start + 1 };
	}
	return undefined;
}

async function listRuns(env: Env): Promise<string[]> {
	// R2 list with delimiter to get per-run subdirs.
	const prefix = `${env.R2_RUNS_PREFIX}/`;
	const out: string[] = [];
	let cursor: string | undefined;
	for (let i = 0; i < 10; i++) {
		const listing = await env.R2.list({ prefix, delimiter: '/', cursor });
		for (const p of listing.delimitedPrefixes) {
			// p looks like 'tomat/runs/<id>/'
			const id = p.slice(prefix.length).replace(/\/$/, '');
			if (id) out.push(id);
		}
		if (!listing.truncated) break;
		cursor = listing.cursor;
	}
	return out.sort();
}

export default {
	async fetch(req: Request, env: Env, _ctx: ExecutionContext): Promise<Response> {
		if (req.method === 'OPTIONS') {
			return new Response(null, { status: 204, headers: corsHeaders(env) });
		}
		if (req.method !== 'GET' && req.method !== 'HEAD') {
			return new Response('Method not allowed', {
				status: 405,
				headers: corsHeaders(env),
			});
		}

		const url = new URL(req.url);
		const path = url.pathname;

		if (path === '/health' || path === '/api/health') {
			return jsonResponse({ ok: true }, env);
		}

		if (path === '/api/runs') {
			const runs = await listRuns(env);
			return jsonResponse({ runs, count: runs.length }, env);
		}

		if (path === '/api/iris-state.json') {
			// Static R2 object updated out-of-band by `tomat iris sync`.
			return serveR2Object(req, env, 'tomat/iris-state.json');
		}

		// /api/runs/:id/<file>
		const runFileMatch = path.match(/^\/api\/runs\/([^/]+)\/(raw\.parquet|manifest\.json)$/);
		if (runFileMatch) {
			const [, runId, file] = runFileMatch;
			const key = `${env.R2_RUNS_PREFIX}/${runId}/${file}`;
			return serveR2Object(req, env, key);
		}

		return new Response(`Not found: ${path}`, {
			status: 404,
			headers: corsHeaders(env),
		});
	},
} satisfies ExportedHandler<Env>;
