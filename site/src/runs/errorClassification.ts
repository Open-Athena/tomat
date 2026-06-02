// Classify an iris attempt's `error` string into a short human-readable cause.
//
// Iris's per-attempt error is typically of the form
// `"Exit code: <N>. stderr: <last lines of stderr>"`. We do a small regex
// pass over the stderr tail to bucket known failure modes — the dashboard
// uses this to caption every `trainer_started` row with WHY that attempt
// died, instead of leaving the user staring at a wall of "trainer_started"
// timestamps with no explanation.
//
// The standing-rule motivation is in specs/45-dashboard-tz11-surfacing.md:
// when a parent iris job reports RUNNING but the per-task histogram is
// all-pending (= no task currently running) and `failures > 0`, the
// dashboard MUST surface the per-attempt failure mode.
//
// Recognised classes — keep this short and curated. Unknown errors fall
// back to the first 80 chars of the cleaned message — better than blank.
// If a new class shows up >2× across runs, add a regex here.

const ERROR_CLASSIFIERS: Array<{
  re: RegExp
  label: string
}> = [
  // The tz-11 post-fix mode: JAX mesh ValueError when the eval-loop data
  // loader runs jit'd `stack_tree` in a bg producer thread that lost the
  // mesh ContextVar. See specs/done/31-tz11-postmortem.md.
  {
    re: /Received incompatible devices for jitted computation/i,
    label: 'JAX mesh ValueError (eval boundary)',
  },
  // The tz-11 pre-fix mode: trainer imported JAX before PJRT_DEVICE=TPU
  // was exported, JAX picked CPU as default. Same `ValueError` class but
  // distinct call site — only the train step.
  {
    re: /jax\.distributed\.initialize\(\) must be called before any JAX calls/i,
    label: 'jax.distributed not initialised',
  },
  // The tz-10 mode: asyncio socket teardown during worker bootstrap.
  {
    re: /asyncio socket\.send\(\) raised exception/i,
    label: 'asyncio socket teardown (worker startup)',
  },
  // Resource exhaustion.
  {
    re: /\b(OOM|OutOfMemory|Resource exhausted|RESOURCE_EXHAUSTED)\b/i,
    label: 'OOM',
  },
  // Coscheduling cascade — task died because a sibling died first.
  // We surface this distinctly so a reader can tell the trigger attempt
  // from the dragged-along ones.
  {
    re: /Coscheduled sibling/i,
    label: 'cascade (sibling died)',
  },
  // SIGTERM — clean preempt-style termination.
  {
    re: /\bSIGTERM\b|Terminated by signal 15/i,
    label: 'SIGTERM',
  },
  // SIGKILL — hard kill, usually preempt or OOM-killer.
  {
    re: /\bSIGKILL\b|Killed by signal 9/i,
    label: 'SIGKILL',
  },
  // Generic CUDA/XLA compilation failure.
  {
    re: /XLA compilation failed|HLO module/i,
    label: 'XLA compile failure',
  },
  // iris worker bounce — controller lost heartbeat from the worker.
  {
    re: /worker ping threshold exceeded/i,
    label: 'worker ping timeout',
  },
  // Tokenizer / vocab-size mismatches at eval restart.
  {
    re: /Axis vocab has different sizes/i,
    label: 'vocab-size mismatch',
  },
]

/** Pick a short label for the given iris-attempt error string. Returns
 *  `null` if the message is empty / whitespace-only. */
export function classifyErrorMessage(error: string | null | undefined): string | null {
  if (!error) return null
  const trimmed = error.trim()
  if (!trimmed) return null
  for (const { re, label } of ERROR_CLASSIFIERS) {
    if (re.test(trimmed)) return label
  }
  // Fall back to a cleaned first-line snippet. Strip the `Exit code: N.
  // stderr: ` boilerplate so the remaining first-line gets to use the
  // budget. Strip iris's stderr-line timestamp prefix (`I20260601 22:50:56
  // 12345 file.cc:123]`) too.
  const cleaned = trimmed
    .replace(/^Exit code:\s*\d+\.\s*stderr:\s*/, '')
    .replace(/^[IWEF]\d{8}\s+\d{2}:\d{2}:\d{2}\.?\d*\s+\d+\s+[\w./:]+\s*/, '')
  const firstLine = cleaned.split(/\r?\n/, 1)[0] ?? cleaned
  return firstLine.length > 80 ? firstLine.slice(0, 77) + '…' : firstLine
}

/** Just the cleaned first line of an iris error message (no classification). */
export function errorFirstLine(error: string | null | undefined): string | null {
  if (!error) return null
  const trimmed = error.trim()
  if (!trimmed) return null
  const cleaned = trimmed
    .replace(/^Exit code:\s*\d+\.\s*stderr:\s*/, '')
  return cleaned.split(/\r?\n/, 1)[0] ?? cleaned
}
