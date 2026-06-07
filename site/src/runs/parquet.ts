// In-browser parquet reading for the runs dashboard.
//
// Uses hyparquet over an HTTP Range-Request source. Returns a column-oriented
// representation since the wallclock/step plots want per-column arrays anyway.

import { parquetReadObjects, asyncBufferFromUrl } from 'hyparquet'

export type RunHistoryRow = {
  _step: bigint | null
  _timestamp: number | null
  _runtime: number | null
  global_step: bigint | null
  'train/loss': number | null
  'eval/loss': number | null
  'throughput/mfu': number | null
  'throughput/tokens_per_second': number | null
  'throughput/duration': number | null
  /** Cumulative GFLOPs since training started — monotonic, logged per step
   *  by Levanter (`throughput/total_gflops`). Multiply by 1e9 for raw FLOPs.
   *  Added 2026-06-07; older parquets won't have it (the column read just
   *  yields an empty array for those runs). */
  'throughput/total_gflops': number | null
  'eval/mat_nmae/val_200/mean': number | null
  'eval/mat_nmae/val_200/median': number | null
  'eval/mat_nmae/val_200/p99': number | null
  'eval/mat_nemd/val_200/mean': number | null
  'eval/mat_nemd/val_200/median': number | null
  'eval/mat_nemd/val_200/p99': number | null
  'lifecycle/trainer_started': number | null
  'lifecycle/sigterm_received': number | null
  'lifecycle/trainer_finished': number | null
  'cluster/preemptions': number | null
  'cluster/failures': number | null
  'cluster/preempts_delta': number | null
  'cluster/failures_delta': number | null
  'cluster/preempts_per_hour': number | null
  'cluster/elapsed_min': number | null
}

export interface RunHistory {
  rowCount: number
  /** ascending unix-seconds for each row */
  timestamps: (number | null)[]
  /** ascending step for each row */
  steps: (number | null)[]
  /** per-column arrays, sparse (null where the metric wasn't logged) */
  cols: Map<keyof RunHistoryRow, (number | null)[]>
}

/**
 * Fetch + decode a run's parquet from the API.
 *
 * We pick column-oriented arrays out of the rows because every downstream
 * plot wants `xs` and `ys` per metric, not per-row dicts.
 */
export async function fetchRunHistory(url: string): Promise<RunHistory> {
  const file = await asyncBufferFromUrl({ url })
  const rows = (await parquetReadObjects({ file })) as RunHistoryRow[]

  const cols = new Map<keyof RunHistoryRow, (number | null)[]>()
  const timestamps: (number | null)[] = []
  const steps: (number | null)[] = []

  const keys = Object.keys(rows[0] ?? {}) as (keyof RunHistoryRow)[]
  for (const k of keys) cols.set(k, [])

  for (const row of rows) {
    // _step is int64 → bigint from hyparquet; coerce to number (safe under ~2^53).
    const stepRaw = row._step
    const step = stepRaw === null ? null : Number(stepRaw)
    steps.push(step)
    timestamps.push(row._timestamp)
    for (const k of keys) {
      const v = row[k]
      cols.get(k)!.push(
        typeof v === 'bigint' ? Number(v) : (v as number | null),
      )
    }
  }

  return { rowCount: rows.length, timestamps, steps, cols }
}

/** Concatenate a list of `RunHistory`s end-to-end into one logical
 *  history — used to render a child run together with its parent(s) on
 *  the detail page (parent + child = full lineage on a single plot).
 *
 *  Caller passes the histories in ancestor-first order (oldest segment
 *  first, then each child in turn). Output columns are the UNION of all
 *  input columns (missing values per-row fill as `null`), and `rowCount`
 *  is the total — so the consumer (WallclockPlot) sees one contiguous
 *  history just as if it had been logged by a single run.
 *
 *  Two rows are not de-duplicated even if a parent's last `_step` ==
 *  child's first; the child's first row typically sits at parent's last
 *  step + 1 (Levanter increments before logging), so de-dup would be a
 *  no-op in practice. */
export function concatHistories(parts: RunHistory[]): RunHistory {
  if (parts.length === 0) {
    return { rowCount: 0, timestamps: [], steps: [], cols: new Map() }
  }
  if (parts.length === 1) return parts[0]
  // Union of column keys across every part.
  const allKeys = new Set<keyof RunHistoryRow>()
  for (const p of parts) for (const k of p.cols.keys()) allKeys.add(k)
  const cols = new Map<keyof RunHistoryRow, (number | null)[]>()
  for (const k of allKeys) cols.set(k, [])
  const timestamps: (number | null)[] = []
  const steps: (number | null)[] = []
  let rowCount = 0
  for (const p of parts) {
    rowCount += p.rowCount
    timestamps.push(...p.timestamps)
    steps.push(...p.steps)
    for (const k of allKeys) {
      const col = cols.get(k)!
      const src = p.cols.get(k)
      if (src) {
        col.push(...src)
      } else {
        // Pad this part's slot with `null` so per-row indexing stays aligned.
        for (let i = 0; i < p.rowCount; i++) col.push(null)
      }
    }
  }
  return { rowCount, timestamps, steps, cols }
}
