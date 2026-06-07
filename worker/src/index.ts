/**
 * tomat-runs-api — CFW backing tomat.oa.dev/runs.
 *
 * Endpoints (all read-only, public for now):
 *   GET  /api/runs-snapshot.json         — aggregated: runs index + all manifests + iris state + modal state + evals indexes + per-run cost.msrp_usd, in one shot
 *   GET  /api/runs                       — list of synced run ids
 *   GET  /api/runs/:id/manifest.json     — per-run metadata (config, summary, history range)
 *   GET  /api/runs/:id/raw.parquet       — full history parquet
 *   GET  /api/runs/:id/eval.json         — per-step mat-NMAE/NEMD series (both mat-sets)
 *   GET  /api/runs/:id/evals             — eval-records index (spec 43): [{key, fired_at, state}]
 *   GET  /api/runs/:id/evals/:key        — one eval record (spec 43, Phase A)
 *   GET  /api/runs/:id/cost.json         — MSRP-equivalent compute estimate (spec 46, Phase A: TPU only)
 *   GET  /api/iris-state.json            — iris snapshot (synced by tomat iris sync)
 *   GET  /api/iris-attempts/:label.json  — per-task per-attempt history (death events) for one training run
 *   GET  /api/voxel-corr/:label.json     — voxel-position corr-matrix sidecar (int8 scale, 1D curve, blob_format)
 *   GET  /api/voxel-corr/:label.bin.gzip — gzipped int8 corr-matrix blob (upper-triangle)
 *   GET  /api/modal-state.json           — Modal app + function-call snapshot (synced by tomat modal sync)
 *   GET  /api/pending-fires.json         — Modal-spawned training fires not yet wandb-manifested (synced by tomat modal pending-fire add)
 *   GET  /api/files/list?prefix=&cursor= — generic R2 list (under FILES_PREFIX allow-list)
 *   GET  /api/files/get?path=…           — generic R2 get with Range support
 *   GET  /health
 *
 * Backed by R2 `openathena/tomat/runs/<id>/{raw.parquet,manifest.json,eval.json,evals/*.json}`,
 * populated out-of-band by `tomat runs sync` + `tomat evals sync` + `tomat evals backfill`
 * (will become an on-demand pull in a later phase — see specs/23-runs-dashboard.md
 * and specs/43-run-eval-relation.md).
 *
 * The `/api/files/*` endpoints expose the broader `tomat/` sub-tree of the
 * `openathena` bucket (runs/, eval/, codecs/, tokenized/, …) for the
 * `@rdub/file-tree`-powered `/files` route on the site. Reuses the same R2
 * binding; the allow-list (`FILES_PREFIX`) keeps consumers from poking at
 * other tenants' data in the shared bucket.
 */

export interface Env {
	R2: R2Bucket;
	CORS_ORIGIN: string;
	R2_RUNS_PREFIX: string;
	/** Allow-list root for /api/files/*. Any list/get outside this prefix is
	 *  rejected. Single string for now (the bucket has one tomat sub-tree);
	 *  if we ever need multiple roots, switch to comma-split + array. */
	FILES_PREFIX: string;
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

async function serveR2Object(
	req: Request,
	env: Env,
	key: string,
	cacheControl: string = 'public, max-age=60',
): Promise<Response> {
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
	headers.set('Cache-Control', cacheControl);
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

/** Eval-records index entry (spec 43 §"Where the record lives"). Each entry
 *  is a pointer to a full record file under `runs/<run>/evals/<key>.json`. */
interface EvalIndexEntry {
	key: string;        // `<step>-<set>-<mode>` or `<step>-<set>-<mode>-task<i>`
	fired_at: string;   // ISO-8601 (≈ GCS object mtime for backfill records)
	state: string;      // pending | running | succeeded | failed | killed
}

/** Tiny cost summary inlined into `runs[i]`. Full breakdown lives in
 *  `runs/<id>/cost.json` (lazy-fetched by the run-detail page on hover).
 *  Sized small so the snapshot doesn't bloat — first-paint chip only. */
interface CostSummary {
	msrp_usd: number;
	is_complete: boolean;
	pricing_table_version: string;
	has_modal_pending: boolean;
}

/** What `runs[id]` carries in the aggregated snapshot. Loosely-typed because
 *  the manifest is the existing `RunManifest` shape (defined in the site's
 *  `api.ts`); we just need to splice in optional `evals` + `cost` fields. */
interface RunSnapshotEntry {
	evals?: EvalIndexEntry[];
	cost?: CostSummary;
	[key: string]: unknown;
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
	runs: Record<string, RunSnapshotEntry | null>;
	iris: unknown | null;
	modal: unknown | null;
	pending_fires: unknown | null;
}> {
	const runIds = await listRuns(env);

	// Fan out manifest.json + evals/index.json reads in parallel. ~50 small
	// R2 GETs in parallel is well within the Worker's subrequest budget
	// (50/req soft cap; 1000/req hard cap on paid plans) and finishes in
	// roughly one R2 round-trip.
	const manifestKeys = runIds.map((id) => `${env.R2_RUNS_PREFIX}/${id}/manifest.json`);
	const evalsIndexKeys = runIds.map((id) => `${env.R2_RUNS_PREFIX}/${id}/evals/index.json`);
	const costKeys = runIds.map((id) => `${env.R2_RUNS_PREFIX}/${id}/cost.json`);
	const [manifestResults, evalsIndexResults, costResults, irisObj, modalObj, pendingFiresObj] = await Promise.all([
		Promise.all(
			manifestKeys.map(async (k): Promise<unknown | null> => {
				const obj = await env.R2.get(k);
				if (!obj) return null;
				try {
					return await obj.json();
				} catch {
					return null;
				}
			}),
		),
		Promise.all(
			evalsIndexKeys.map(async (k): Promise<EvalIndexEntry[] | null> => {
				const obj = await env.R2.get(k);
				if (!obj) return null;
				try {
					const parsed = await obj.json();
					return Array.isArray(parsed) ? (parsed as EvalIndexEntry[]) : null;
				} catch {
					return null;
				}
			}),
		),
		Promise.all(
			costKeys.map(async (k): Promise<CostSummary | null> => {
				// Spec 46: inline only the tiny summary fields — full breakdown
				// is lazy-fetched on the run-detail page's chip tooltip.
				const obj = await env.R2.get(k);
				if (!obj) return null;
				try {
					const parsed = await obj.json() as {
						msrp_usd?: number;
						is_complete?: boolean;
						pricing_table_version?: string;
						breakdown?: { kind?: string; msrp_usd?: number | null }[];
					};
					if (typeof parsed.msrp_usd !== 'number') return null;
					// `has_modal_pending` should mean "there is a Modal segment we
					// HAVEN'T priced yet" — i.e. a placeholder with null msrp_usd.
					// Modal segments with a real msrp_usd (Phase B) are no longer
					// "pending"; the chip should show `$X MSRP`, not "Modal MSRP TBD".
					const hasModalPending = Array.isArray(parsed.breakdown)
						&& parsed.breakdown.some((b) => b?.kind === 'modal' && b?.msrp_usd == null);
					return {
						msrp_usd: parsed.msrp_usd,
						is_complete: !!parsed.is_complete,
						pricing_table_version: String(parsed.pricing_table_version ?? ''),
						has_modal_pending: hasModalPending,
					};
				} catch {
					return null;
				}
			}),
		),
		env.R2.get('tomat/iris-state.json'),
		env.R2.get('tomat/modal-state.json'),
		env.R2.get('tomat/pending-fires.json'),
	]);

	const runs: Record<string, RunSnapshotEntry | null> = {};
	for (let i = 0; i < runIds.length; i++) {
		const manifest = manifestResults[i] as RunSnapshotEntry | null;
		const evals = evalsIndexResults[i];
		const cost = costResults[i];
		// Drop entries whose manifest is missing the load-bearing trio
		// (`run`, `history`, `summary`). About 10 stale entries in R2 (older v2
		// P14 runs + a few partials) have only `cost.json` populated; without
		// the manifest fields the dashboard can't render them and every consumer
		// that reaches into sub-objects crashes. Emit `null` so the FE's
		// existing null-filter drops them cleanly.
		const manifestValid = manifest
			&& manifest.run != null
			&& manifest.history != null
			&& manifest.summary != null;
		if (!manifestValid) {
			runs[runIds[i]] = null;
			continue;
		}
		const entry: RunSnapshotEntry = manifest;
		if (evals && evals.length > 0) entry.evals = evals;
		if (cost) entry.cost = cost;
		runs[runIds[i]] = entry;
	}

	let iris: unknown | null = null;
	if (irisObj) {
		try {
			iris = await irisObj.json();
		} catch {
			iris = null;
		}
	}
	let modal: unknown | null = null;
	if (modalObj) {
		try {
			modal = await modalObj.json();
		} catch {
			modal = null;
		}
	}
	let pending_fires: unknown | null = null;
	if (pendingFiresObj) {
		try {
			pending_fires = await pendingFiresObj.json();
		} catch {
			pending_fires = null;
		}
	}

	return {
		synced_at: new Date().toISOString(),
		count: runIds.length,
		runs,
		iris,
		modal,
		pending_fires,
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

/** Reject any path outside the FILES_PREFIX allow-list. The allow-list is a
 *  single root (e.g. `tomat/`) for now — see `Env.FILES_PREFIX`. Empty
 *  `FILES_PREFIX` means "whole bucket"; we don't expose that by default. */
function isFilesPrefixAllowed(env: Env, keyOrPrefix: string): boolean {
	const root = env.FILES_PREFIX;
	if (!root) return false;
	// Both empty prefix → "root listing" — allow, the lister will scope to
	// the root prefix anyway.
	if (keyOrPrefix === '' || keyOrPrefix === root) return true;
	return keyOrPrefix.startsWith(root);
}

interface FilesEntry {
	/** Store-relative path. Directories end with `/`. */
	key: string;
	/** Bytes; absent for directories. */
	size?: number;
	/** ISO-8601; absent if the store doesn't track it. */
	lastModified?: string;
	isDir: boolean;
}

interface FilesListResult {
	entries: FilesEntry[];
	cursor?: string;
}

/** Generic R2 list with delimiter, emitting the file-tree HTTP shape that
 *  `HttpStore` consumes. Edge-caches per (prefix, cursor) tuple for 60s —
 *  R2 list isn't free and these are mostly static blob trees. */
async function handleFilesList(req: Request, env: Env): Promise<Response> {
	const url = new URL(req.url);
	let prefix = url.searchParams.get('prefix') ?? '';
	const cursor = url.searchParams.get('cursor') ?? undefined;
	const limitS = url.searchParams.get('limit');
	const limit = limitS ? Math.min(1000, Math.max(1, parseInt(limitS, 10) || 100)) : undefined;

	// Empty prefix is meaningful: the file-tree UI's "root" listing. Scope
	// to FILES_PREFIX so consumers see the tomat sub-tree as the apparent
	// bucket root (no peeking at other openathena tenants).
	if (prefix === '') {
		prefix = env.FILES_PREFIX;
	} else if (!isFilesPrefixAllowed(env, prefix)) {
		return new Response('Forbidden: prefix outside FILES_PREFIX', {
			status: 403,
			headers: corsHeaders(env),
		});
	}

	const listing = await env.R2.list({
		prefix,
		delimiter: '/',
		...(cursor ? { cursor } : {}),
		...(limit ? { limit } : {}),
	});

	const entries: FilesEntry[] = [];
	// delimitedPrefixes = sub-"directories"; each ends with `/`.
	for (const p of listing.delimitedPrefixes ?? []) {
		entries.push({ key: p, isDir: true });
	}
	for (const o of listing.objects ?? []) {
		// Skip the "directory marker" objects R2 sometimes has at the prefix
		// itself (key === prefix). These confuse the tree UI.
		if (o.key === prefix) continue;
		entries.push({
			key: o.key,
			size: o.size,
			lastModified: o.uploaded instanceof Date ? o.uploaded.toISOString() : undefined,
			isDir: false,
		});
	}

	const out: FilesListResult = { entries };
	if (listing.truncated && listing.cursor) {
		out.cursor = listing.cursor;
	}

	return new Response(JSON.stringify(out), {
		headers: {
			'Content-Type': 'application/json',
			'Cache-Control': 'public, max-age=60',
			...corsHeaders(env),
		},
	});
}

/** Best-effort content-type from extension. R2's `httpMetadata.contentType`
 *  wins when set; this is the fallback for blobs uploaded without one
 *  (most of our `tomat runs sync` output). */
function contentTypeForKey(key: string): string {
	const i = key.lastIndexOf('.');
	const ext = i >= 0 ? key.slice(i + 1).toLowerCase() : '';
	switch (ext) {
		case 'json': return 'application/json';
		case 'md': case 'markdown': return 'text/markdown; charset=utf-8';
		case 'txt': case 'log': return 'text/plain; charset=utf-8';
		case 'csv': return 'text/csv; charset=utf-8';
		case 'tsv': return 'text/tab-separated-values; charset=utf-8';
		case 'html': case 'htm': return 'text/html; charset=utf-8';
		case 'js': return 'application/javascript';
		case 'png': return 'image/png';
		case 'jpg': case 'jpeg': return 'image/jpeg';
		case 'gif': return 'image/gif';
		case 'svg': return 'image/svg+xml';
		case 'webp': return 'image/webp';
		case 'mp4': return 'video/mp4';
		case 'webm': return 'video/webm';
		case 'mp3': return 'audio/mpeg';
		case 'wav': return 'audio/wav';
		case 'parquet': case 'pqt': return 'application/octet-stream';
		case 'zip': return 'application/zip';
		case 'pdf': return 'application/pdf';
		case 'ipynb': return 'application/x-ipynb+json';
		default: return 'application/octet-stream';
	}
}

/** Serve an R2 object under the FILES_PREFIX allow-list. Like
 *  `serveR2Object`, but uses extension-derived content-type when R2 has no
 *  stored `httpMetadata.contentType`, and applies cache-control tuned for
 *  static blobs (long for `*.parquet`, shorter for `*.json` which may
 *  change as runs sync). */
async function serveR2FilesObject(req: Request, env: Env, key: string): Promise<Response> {
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
	if (!headers.get('Content-Type')) {
		headers.set('Content-Type', contentTypeForKey(key));
	}
	headers.set('etag', obj.httpEtag);
	headers.set('Accept-Ranges', 'bytes');
	// Cache parquet blobs longer (immutable per run id); json/log/manifest
	// shorter since they can be re-synced.
	const longTtl = /\.(parquet|pqt|zip|png|jpg|jpeg|webp|mp4|webm|mp3|wav|svg|pdf)$/i.test(key);
	headers.set('Cache-Control', longTtl ? 'public, max-age=86400' : 'public, max-age=60');
	const totalSize = obj.size;
	let status = 200;
	if (r2Range) {
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
	return new Response(obj.body, { status, headers });
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

		if (path === '/api/modal-state.json') {
			// Static R2 object updated out-of-band by `tomat modal sync`.
			// Modal app + function-call snapshot — feeds the Modal-state
			// badge on /runs (replaces the wandb-session-state fallback for
			// runs whose iris job slot is empty).
			return serveR2Object(req, env, 'tomat/modal-state.json');
		}

		if (path === '/api/pending-fires.json') {
			// Modal-spawned training fires that wandb hasn't manifested yet.
			// Written by `tomat modal pending-fire add` right after
			// `fn.spawn(...)`, GC'd after the wandb manifest lands. Powers
			// placeholder cards on /runs during the pre-wandb window.
			return serveR2Object(req, env, 'tomat/pending-fires.json');
		}

		// /api/iris-attempts/<label>.json — per-task attempt history sidecar,
		// emitted alongside iris-state.json by the same `tomat iris sync` pass.
		// Drives the death-cause vlines + % training chip on /runs/<label>.
		const attemptsMatch = path.match(/^\/api\/iris-attempts\/([^/]+)\.json$/);
		if (attemptsMatch) {
			const [, label] = attemptsMatch;
			return serveR2Object(req, env, `tomat/iris-attempts/${label}.json`);
		}

		// /api/voxel-corr/<label>.{json,bin.gzip} — per-run voxel-position
		// correlation blob + sidecar (emitted by `tomat analyze voxel-corr-blob
		// --r2`). Effectively immutable per label, so we cache long (1h at the
		// edge; the underlying R2 object's etag handles invalidation).
		const voxelCorrMatch = path.match(/^\/api\/voxel-corr\/([^/]+)\.(json|bin\.gzip)$/);
		if (voxelCorrMatch) {
			const [, label, ext] = voxelCorrMatch;
			return serveR2Object(
				req, env, `tomat/voxel-corr/${label}.${ext}`,
				'public, max-age=3600',
			);
		}

		// /api/runs/:id/<file>
		const runFileMatch = path.match(/^\/api\/runs\/([^/]+)\/(raw\.parquet|manifest\.json|eval\.json|cost\.json)$/);
		if (runFileMatch) {
			const [, runId, file] = runFileMatch;
			const key = `${env.R2_RUNS_PREFIX}/${runId}/${file}`;
			return serveR2Object(req, env, key);
		}

		// /api/runs/:id/evals — eval-records index (spec 43). Returns the
		// `index.json` sidecar; the dashboard hot-cache snapshot already
		// inlines this, but exposing it standalone lets the detail page
		// re-fetch independently of the 30s snapshot edge-cache.
		const evalsIndexMatch = path.match(/^\/api\/runs\/([^/]+)\/evals\/?$/);
		if (evalsIndexMatch) {
			const [, runId] = evalsIndexMatch;
			const key = `${env.R2_RUNS_PREFIX}/${runId}/evals/index.json`;
			// 404 → empty array (no records yet, not an error).
			const obj = await env.R2.get(key);
			if (!obj) return jsonResponse([], env);
			return serveR2Object(req, env, key);
		}

		// /api/runs/:id/evals/:key — one eval record (spec 43). Key is
		// `<step>-<set>-<mode>` optionally with `-task<i>`. Validate the key
		// shape so we don't expose arbitrary R2-key traversal under
		// `runs/<id>/evals/`.
		const evalRecordMatch = path.match(/^\/api\/runs\/([^/]+)\/evals\/([A-Za-z0-9._-]+)$/);
		if (evalRecordMatch) {
			const [, runId, rawKey] = evalRecordMatch;
			// Strip a trailing `.json` so both `key` and `key.json` work.
			const key = rawKey.endsWith('.json') ? rawKey.slice(0, -5) : rawKey;
			const r2Key = `${env.R2_RUNS_PREFIX}/${runId}/evals/${key}.json`;
			return serveR2Object(req, env, r2Key);
		}

		if (path === '/api/files/list') {
			return handleFilesList(req, env);
		}

		if (path === '/api/files/get') {
			const key = url.searchParams.get('path') ?? '';
			if (!key) {
				return new Response('path required', { status: 400, headers: corsHeaders(env) });
			}
			if (!isFilesPrefixAllowed(env, key)) {
				return new Response('Forbidden: path outside FILES_PREFIX', {
					status: 403,
					headers: corsHeaders(env),
				});
			}
			return serveR2FilesObject(req, env, key);
		}

		return new Response(`Not found: ${path}`, {
			status: 404,
			headers: corsHeaders(env),
		});
	},
} satisfies ExportedHandler<Env>;
