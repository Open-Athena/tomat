/**
 * tomat-runs-api — CFW backing tomat.oa.dev/runs.
 *
 * Endpoints (all read-only, public for now):
 *   GET  /api/runs-snapshot.json         — aggregated: runs index + all manifests + iris state, in one shot
 *   GET  /api/runs                       — list of synced run ids
 *   GET  /api/runs/:id/manifest.json     — per-run metadata (config, summary, history range)
 *   GET  /api/runs/:id/raw.parquet       — full history parquet
 *   GET  /api/runs/:id/eval.json         — per-step mat-NMAE/NEMD series (both mat-sets)
 *   GET  /api/iris-state.json            — iris snapshot (synced by tomat iris sync)
 *   GET  /health
 *
 * Backed by R2 `openathena/tomat/runs/<id>/{raw.parquet,manifest.json,eval.json}`,
 * populated out-of-band by `tomat runs sync` + `tomat evals sync` (will become
 * an on-demand pull in a later phase — see specs/23-runs-dashboard.md).
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

/**
 * Aggregated snapshot endpoint — assembles the full runs-dashboard payload in
 * ONE response so the frontend doesn't have to fan out N per-run requests.
 *
 * Replaces the old `/api/runs` + N×`/api/runs/:id/manifest.json` + 1×iris poll
 * pattern with a single edge-cached endpoint. Runs in the index that don't
 * have a manifest.json yet (e.g. just created, never synced) appear with
 * `manifest: null` so the frontend can skip them — no more 404 fanout.
 *
 * Edge cache TTL is 30s: at 3 polls/min from each client, an N-client tab
 * fleet still only hits R2 once per 30s window. R2 list + ~50 parallel
 * `r2.get()` calls is fast (~200-500ms cold) but not free, so we cache.
 *
 * Cache key uses the request URL (no auth / no per-user differentiation), so
 * all clients share one cached body.
 */
async function buildRunsSnapshot(env: Env): Promise<{
	synced_at: string;
	count: number;
	runs: Record<string, unknown | null>;
	iris: unknown | null;
}> {
	const runIds = await listRuns(env);

	// Fan out manifest.json reads in parallel. ~50 small R2 GETs in parallel
	// is well within the Worker's subrequest budget (50/req soft cap; 1000/req
	// hard cap on paid plans) and finishes in roughly one R2 round-trip.
	const manifestKeys = runIds.map((id) => `${env.R2_RUNS_PREFIX}/${id}/manifest.json`);
	const manifestResults = await Promise.all(
		manifestKeys.map(async (k): Promise<unknown | null> => {
			const obj = await env.R2.get(k);
			if (!obj) return null;
			try {
				return await obj.json();
			} catch {
				return null;
			}
		}),
	);

	const runs: Record<string, unknown | null> = {};
	for (let i = 0; i < runIds.length; i++) {
		runs[runIds[i]] = manifestResults[i];
	}

	// Inline the iris snapshot too — same R2 bucket, one more parallel GET
	// would also work but this is already serial-after-the-fanout and cheap.
	let iris: unknown | null = null;
	const irisObj = await env.R2.get('tomat/iris-state.json');
	if (irisObj) {
		try {
			iris = await irisObj.json();
		} catch {
			iris = null;
		}
	}

	return {
		synced_at: new Date().toISOString(),
		count: runIds.length,
		runs,
		iris,
	};
}

/** TTL for the snapshot edge cache. 30s matches the existing per-manifest
 *  active-run poll cadence; the frontend's `refetchInterval` is independent
 *  (TanStack-side) so this only governs how often any one Worker call has to
 *  do real R2 work. */
const SNAPSHOT_CACHE_TTL = 30;

function snapshotResponse(snapshot: unknown, env: Env): Response {
	return new Response(JSON.stringify(snapshot), {
		headers: {
			'Content-Type': 'application/json',
			// Workers Cache API honors this for its own TTL.
			'Cache-Control': `public, max-age=${SNAPSHOT_CACHE_TTL}`,
			...corsHeaders(env),
		},
	});
}

async function handleRunsSnapshot(req: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
	const cache = caches.default;
	// Use the canonical URL (path only, no host) for the cache key so a future
	// host swap (e.g. moving the API under tomat.oa.dev) doesn't fragment the
	// cache. The Workers Cache API requires an absolute URL, so we synthesize
	// one off the request's origin.
	const cacheKey = new Request(new URL('/api/runs-snapshot.json', new URL(req.url)).toString(), { method: 'GET' });

	const cached = await cache.match(cacheKey);
	if (cached) {
		// Stale-while-revalidate: if the cached body is past its half-life,
		// kick off a background refresh. The current request still gets the
		// cached copy (instant) but the next one will see fresher data.
		// `Age` is set by CF's edge cache; if missing (some test harnesses)
		// we just always-serve-while-fresh.
		const ageHeader = cached.headers.get('Age');
		const age = ageHeader ? parseInt(ageHeader, 10) : 0;
		if (age >= SNAPSHOT_CACHE_TTL / 2) {
			ctx.waitUntil(refreshSnapshotCache(cache, cacheKey, env));
		}
		return cached;
	}

	// Cold path: compute + store.
	const snapshot = await buildRunsSnapshot(env);
	const resp = snapshotResponse(snapshot, env);
	// Edge-cache a clone (the response we return is consumed by the client).
	ctx.waitUntil(cache.put(cacheKey, resp.clone()));
	return resp;
}

/** Recompute the snapshot and update the edge cache. Called from
 *  `ctx.waitUntil` to refresh stale entries without blocking the request. */
async function refreshSnapshotCache(cache: Cache, cacheKey: Request, env: Env): Promise<void> {
	const snapshot = await buildRunsSnapshot(env);
	await cache.put(cacheKey, snapshotResponse(snapshot, env));
}

export default {
	async fetch(req: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
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

		if (path === '/api/runs-snapshot.json') {
			return handleRunsSnapshot(req, env, ctx);
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
		const runFileMatch = path.match(/^\/api\/runs\/([^/]+)\/(raw\.parquet|manifest\.json|eval\.json)$/);
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
