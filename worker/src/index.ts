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

async function serveR2Object(env: Env, key: string): Promise<Response> {
	const obj = await env.R2.get(key);
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
	for (const [k, v] of Object.entries(corsHeaders(env))) {
		headers.set(k, v as string);
	}
	return new Response(obj.body, { headers });
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
		if (req.method !== 'GET') {
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

		// /api/runs/:id/<file>
		const runFileMatch = path.match(/^\/api\/runs\/([^/]+)\/(raw\.parquet|manifest\.json)$/);
		if (runFileMatch) {
			const [, runId, file] = runFileMatch;
			const key = `${env.R2_RUNS_PREFIX}/${runId}/${file}`;
			return serveR2Object(env, key);
		}

		return new Response(`Not found: ${path}`, {
			status: 404,
			headers: corsHeaders(env),
		});
	},
} satisfies ExportedHandler<Env>;
