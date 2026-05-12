// In-browser parquet reading for the runs dashboard.
//
// Uses hyparquet over an HTTP Range-Request source. Returns a column-oriented
// representation since the wallclock/step plots want per-column arrays anyway.

import { parquetReadObjects, asyncBufferFromUrl } from 'hyparquet'

export type RunHistoryRow = {
  _step: bigint | null
  _timestamp: number | null
  _runtime: number | null
  'train/loss': number | null
  'eval/loss': number | null
  'throughput/mfu': number | null
  'throughput/tokens_per_second': number | null
  'throughput/duration': number | null
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
