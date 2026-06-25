# Region-strategy defaults for `tomat train`

**Status**: draft — sequenced after spec 61 P2 lands (fires-as-records) and after `tomat train`
absorbs the 4 missing MaskGIT knobs that block .sh-script migration (task #341).

**Triggering pain**:

- Every from-scratch run has had to pick *one* region up-front, even when caches +
  codecs already exist in 3+ regions. We pay capacity-block delays (hours) waiting for a
  pinned region when iris could have scheduled us elsewhere immediately.
- Every resume fire has carried a region in human-author shell-script knobledge: the .sh
  hardcodes `--zone us-east5-b` because *the author remembered* the parent's region. Memory
  failures here have caused cross-region ckpt egress (see [[cross-region-eval-egress]]) and
  inadvertent capacity battles ("just resume on v6e" → user types `us-east5-b` → that pool
  is empty for 6h).
- We have all the data-mirroring tooling (`tomat cache mirror`, `tomat ckpt mirror`,
  `--share-cache` zone-detect) but no single CLI gesture that wires them together as a
  pre-flight + iris-fire pair.

Spec 61 codified *what* a fire is and *that* it has a parent. This spec codifies *where*
it gets to run and *what data has to be mirrored* before it does.

## Pain points the spec addresses

1. **One-region pin is a default by accident.** Every script picks a single `--zone` because
   the author thought about that region first. Multi-region is supported by iris (see §3.1)
   but exposed nowhere in `tomat train`.
2. **Resumes silently inherit the wrong region defaults.** Memory says "pin eval to ckpt
   region" — but `tomat train` resumes don't enforce or surface this. The .sh author
   types `--zone X` and we hope X is right.
3. **Pre-flight data mirroring is an opt-in chore.** Codecs, caches, and parent ckpts must be
   region-local for performance (and sometimes correctness — see
   [[cross-region-codec-mirror]]: a missing codec mirror crashes the trainer at step 0).
   We have CLI tools for each but no glue.
4. **Capacity-block lookups are manual.** `iris avail` exists; nothing wires its output into
   a fire decision. The .sh author manually checks `iris avail`, picks a region, types the
   zone — repeating the same human work N times per week.

## Goals

- `tomat train` accepts `--region-strategy {any,auto,explicit}` with a sensible
  per-mode default (from-scratch → `any`, resume → `auto`).
- `any` and `auto` perform pre-flight data mirroring transparently (cache, codec, ckpt as
  needed) using the existing `tomat cache mirror` / `tomat ckpt mirror` machinery.
- `any` issues a multi-region iris fire (`--region X --region Y --region Z`); iris picks
  among eligible regions. The trainer's existing zone-detect handles the rest.
- `auto` probes `iris avail` once, picks the best region by capacity, pre-mirrors data to
  that one region, fires pinned to that region.
- `explicit` preserves the current "the author knows what they want" path.
- Every fire's R2 manifest (spec 61 §2.1) records the chosen strategy + iris-avail snapshot
  at decision time, so post-mortems can answer "why did this land in eu-west4?".

## Non-goals

- No new GCS storage class management. Mirroring uses the same regional buckets we already
  populate (`marin-us-east5`, `marin-eu-west4`, `marin-us-central2`).
- No iris-side contributions. iris already supports `--region` (repeatable); we use that.
- No predictive capacity model. We probe at fire-time and act on the snapshot. "Capacity is
  claimed in the gap" is already handled by `--queue` retry.
- No Modal-side region strategy. Modal containers don't expose region selection to the user
  in the same way; this spec is iris-only. Modal fires set `region_strategy: "n/a"` in their
  manifest.

## Status quo: what's already in place

The pieces:

- **`iris job run --region X --region Y --region Z`** — iris supports repeatable `--region`
  flags; the scheduler picks among eligible regions based on pool availability. Verified in
  `marin/lib/iris/src/iris/cli/job.py` line 817:
  ```python
  @click.option("--region", multiple=True, help="Restrict to region(s) ...")
  ```
- **`tomat cache mirror -l <label>`** — mirrors a Levanter cache across the canonical
  regional buckets (us-east5, eu-west4, us-central2). One-time per label; idempotent.
- **`tomat ckpt mirror <label> [--targets ...]`** — mirrors a single step (or all steps) of a
  ckpt across regional buckets. Idempotent.
- **`TOMAT_SHARE_CACHE=1`** (default on, see `tomat.py` line 4282) — trainer detects its
  GCE zone at startup via metadata server and resolves the cache mirror path locally. With
  `share-cache=1` and no `--zone` pin, the trainer already handles "land wherever, find
  local cache."
- **Trainer writes ckpts to its local-region bucket** (#210 done). So a resume from such
  a ckpt requires reading from the region the parent landed in — not always the bucket the
  fire was nominally configured with.
- **`tomat train --queue`** — blocks on capacity, retries when iris rejects. Composes
  naturally with multi-region: queue across all eligible regions.

The pieces NOT in place:

- **Resume bucket auto-resolution.** Trainer reads ckpt from `BUCKET/results/<label>/...`
  where `BUCKET` is set by the fire. If the parent's last segment landed in eu-west4 but
  the resume specifies `BUCKET=gs://marin-us-east5/tomat`, the trainer pays cross-region
  egress on every restart load. This is a pre-existing bug worth a separate fix; this spec
  works around it by either mirroring ckpts to *every* candidate region (in `any` mode) or
  pinning the resume to the parent's region (in `auto` mode).
- **Codec mirroring.** Most codecs live at `gs://marin-eu-west4/tomat/codecs/`. A us-east5
  worker reads them cross-region at startup — small (~MB) but un-mirrored. This spec adds a
  per-region codec mirror to the `any` pre-flight.

## The strategies

### `any` — "fire to any region iris will accept"

**Default for from-scratch fires.**

Known target regions (from `_CACHE_TARGETS` in `tomat:5015`):

| region        | bucket                          | TPU pools served                   |
| ------------- | ------------------------------- | ---------------------------------- |
| eu-west4      | `gs://marin-eu-west4/tomat`     | canonical source; all TPU pools    |
| us-east5      | `gs://marin-us-east5/tomat`     | us-east5-a (v5p), us-east5-b (v6e) |
| us-central1   | `gs://marin-us-central1/tomat`  | us-central1-a (v5p)                |
| us-east1      | `gs://marin-us-east1/tomat`     | us-east1-d (v6e)                   |

Survey (2026-06-23) — current cache state across regions:

| label                         | eu-west4 | us-east5 | us-central1 | us-east1 |
| ----------------------------- | -------- | -------- | ----------- | -------- |
| train-full-v3 (TS0)           | ✅       | ✅       | ✅          | ✅       |
| train-full-v3-shard{1,2,3}    | ✅       | ✅       | shard1 only | shard1 only |
| train-full-lmq-v2-lat         | ✅       | ❌       | ❌          | ❌       |

**Important consequence**: not all regions are eligible for every fire. A TS0123-union fire
can only land in eu-west4 OR us-east5 *unless we mirror shards 2-3 to the others first*.

Pre-flight (idempotent — skips if already mirrored):
1. **Cache**: for every distinct `data_label` referenced, check each region for full presence.
   - **Already present in all eligible regions** (common case): no-op.
   - **Missing in some regions**: choose between (a) mirroring before firing, or (b)
     dropping the missing-region from the eligible set. Default = (b) — mirroring tens-of-
     GB shards at fire time can take 30+ minutes; prefer firing fast in an already-mirrored
     region. Caller can pass `--mirror-on-pre-flight` to opt into (a).
2. **Codec**: call (new) `tomat codec mirror <codec-path> --targets <eligible-regions>`.
   Codecs are tiny (~MB); always mirror to all eligible regions (no opt-out).
3. **Ckpt**: from-scratch only, so no parent ckpt to mirror.

Fire:
- iris invocation: `--region <region1> --region <region2> ...` over the eligible region set.
- iris picks among eligible regions based on capacity; trainer's zone-detect handles
  local-cache resolution.

Cost: pre-flight is one-time and idempotent. Steady-state ($/fire) = 0 extra. First-time-
per-label cost (if opted in to `--mirror-on-pre-flight`) = N × cache size (typically
tens of GB per shard) + N × codec size (~MB) in regional GCS storage. Storage is cheap
(~$0.02/GB/mo) — this is rounding error on top of TPU spend.

### `auto` — "pre-flight + pick one region"

**Default for resume fires.**

Pre-flight:
1. Probe `iris avail` for the requested TPU shape.
2. Pick the region with the highest READY count among eligible regions for this fire's
   parent (if `--parent` set, eligible regions = those where the parent has at least one
   ckpt or where the cache is already mirrored).
3. **Cache**: mirror to the picked region if not already present.
4. **Codec**: mirror to the picked region if not already present.
5. **Ckpt**: call `tomat ckpt mirror <parent_label> --targets <picked-region>`. This is the
   load-bearing mirror — without it, the resume reads cross-region on every restart.

Fire:
- iris invocation: `--region <picked>` (single region pin).
- If `--queue` is also set: if the picked region's capacity is claimed in the gap, fall
  through to re-probing on the next queue tick (vs. blocking on the picked region forever).

Why prefer `auto` for resumes:
- Resumes mirror the parent's ckpt — *N*-region mirror is *N* × ckpt size (tens of GB each).
  Worth doing once for a long-running production fire; not worth doing for every smoke.
- Pinning to one region means the in-region ckpt-read on every restart is local; no cross-
  region egress recurring cost.

### `explicit` — "I know what I want"

Caller passes `--region X` or `--zone Y` directly; no pre-flight, no auto-mirror, no probe.
This is the current behavior, retained for:
- Hypothesis tests like `bin5-cont-from-80k-v6e.sh` ("test parent's HW on parent's zone").
- Recovery from a known-bad region picker (override the default if `auto` keeps picking
  the wrong thing).
- Single-region fires where the author has full context.

The caller is responsible for ensuring the data is mirrored to that region. No safety net.

## CLI surface

```
tomat train ... \
  --region-strategy {any,auto,explicit}   # default: any (from-scratch), auto (resume)
  --region X --region Y                    # used only by `explicit` (or as a filter on
                                           # the candidate-region set for any/auto)
  --zone Z                                 # used only by `explicit`
  --skip-pre-flight                        # opt-out for "I just mirrored, don't re-check"
```

Validations:

- `--region-strategy any` + `--zone Z` → error: "use --region-strategy explicit if you want
  to pin a zone".
- `--region-strategy auto` + `--region X --region Y` (multi) → treat as a candidate filter
  ("pick one among X, Y based on availability").
- `--region-strategy explicit` with neither `--region` nor `--zone` → error.

## Fire manifest extension (extends spec 61 §2.1)

```jsonc
{
  // ... existing spec 61 fields ...
  "iris": {
    "job_id": "/ryan/train-mg-...",
    "region_strategy": "any" | "auto" | "explicit",
    "region_eligible": ["us-east5", "eu-west4", "us-central2"],  // what iris was given
    "region_picked": "eu-west4",         // null for `any` (let iris decide)
    "region_avail_snapshot": {           // null for `explicit` (didn't probe)
      "probed_at": "2026-06-23T19:00:00Z",
      "ready": {"us-east5": 0, "eu-west4": 4, "us-central2": 2},
      "tpu_shape": "v6e-16"
    },
    "pre_flight_mirrors": {              // what we actually mirrored
      "cache": {"label": "train-full-v3,...", "to": ["eu-west4"], "skipped_existing": ["us-east5", "us-central2"]},
      "codec": {"path": "gs://marin-eu-west4/tomat/codecs/lmq-v2-16k.npz", "to": ["us-east5", "us-central2"], "skipped_existing": ["eu-west4"]},
      "ckpt":  {"label": "train-mg-kl-bin5-fs-tpu", "step": 80000, "to": ["eu-west4"], "skipped_existing": []}
    }
  }
}
```

This gives the dashboard everything it needs to render the "why was this fire here?" line
on the run-detail page.

## Phasing

### Phase A — CLI surface + manifest extension (1-2 hours)
- Add `--region-strategy` to `tomat train`. Map per-mode default (from-scratch → any,
  resume → auto).
- Extend `_record_train_fire_to_r2` (spec 61 P2 helper) to capture `region_strategy`,
  `region_eligible`, `region_picked`, `region_avail_snapshot` in the manifest.
- Wire `--region-strategy explicit` through to current `--region`/`--zone` plumbing
  (status quo behavior; new field on manifest).
- **Validation only**: `any` and `auto` modes still fall through to a single-region
  fire; this phase just lays the surface so callers can opt in.

### Phase B — `auto` mode pre-flight (2-3 hours)
- Implement `iris avail` probe inside `tomat train`. Pick highest-READY region.
- Wire `tomat cache mirror`, `tomat ckpt mirror` into pre-flight calls (subprocess to the
  existing CLIs — they're well-tested; no need to inline their logic).
- Add `tomat codec mirror <path> --targets ...` (new sub-command; small).
- Emit a one-line summary of mirrored / skipped data before the iris call.
- Default `--region-strategy auto` for `tomat train --resume`.

### Phase C — `any` mode multi-region fire (1-2 hours)
- Pass `--region` repeatedly to iris based on the eligible-region set.
- Pre-flight mirrors to all targets (idempotent — usually no-op).
- Default `--region-strategy any` for `tomat train` from-scratch.
- Re-test the trainer's zone-detect path end-to-end (already done in #99, #210, but worth
  re-verifying with a smoke).

### Phase D — Dashboard surface (optional, after A-C)
- `RunHeaderRich` adds a region badge: "fired with `auto` → eu-west4 (was 0/4/2 READY)".
- Click through to a per-fire iris-avail snapshot table.

## Open issues / unknowns to verify during impl

1. ~~**iris `--region` substring matching**~~ **Resolved**: iris's `--region` is exact-match
   against `ScalingGroup.region`. `--region us-east5` matches all us-east5 zones; passing a
   zone-flavored value (e.g. `us-east5-a`) is rejected with an error suggesting `--zone`
   instead. Confirmed in
   `marin/lib/iris/src/iris/cluster/controller/autoscaler/routing.py:253` (`_looks_like_zone`
   check). The spec's `--region` semantics are correct as drafted.
2. **`tomat ckpt mirror` zone vs region**: today's CLI uses bucket targets, not iris-pool
   regions. The mapping is 1:1 against `_CACHE_TARGETS`'s 4-region table above; we'd need to
   double-check that `ckpt mirror`'s default-targets list is the same set before Phase B
   (it's defined separately in tomat:5185+ — read during Phase B impl).
3. ~~**Cache mirror cost vs frequency**~~ **Surveyed** (2026-06-23): see the cache-state
   table above. Resolution: not every region has every cache. The `any` pre-flight's
   *default* should be (b) "drop the missing-region from the eligible set", with
   `--mirror-on-pre-flight` as opt-in. Mirroring a 50+ GB shard at fire time can block iris
   for 30+ minutes — almost always worse than just firing in an already-mirrored region.
4. **Resume cross-region read fallback**: if `auto` picks region X but X's ckpt mirror is
   stale (parent's most recent step landed elsewhere and hasn't been mirrored yet),
   `auto` should detect and either (a) mirror just that one step, or (b) fall back to the
   parent's last-write region. Phase B impl detail. Recommended: (a), since one-step
   mirroring is fast (~minutes for ~10GB), and the "wherever the parent last landed" picker
   defeats the point of region-strategy ("capacity now" should dominate "convenience now").
5. **`--queue` × `auto`**: re-probe on each queue tick, or stick with the first pick? Should
   be re-probe (capacity in eu-west4 might recover faster than us-east5 — a stale pick
   wastes the queue's benefit). But re-probing means re-mirroring if the new pick differs;
   `--queue` × `auto` × `--mirror-on-pre-flight` could end up mirroring N times across a
   long queue. Solution: cache the mirror state per (label, region) and only mirror when
   the queue picks a new region not previously mirrored to.
6. **Modal manifest field**: spec 61 §2.1 has a `modal:` block; this spec adds `region_*`
   fields to the `iris:` block only. Decision: Modal fires omit the field entirely (no
   `region_strategy` key) rather than carrying a `"n/a"` sentinel. The dashboard renders
   "no region info" for the field's absence; this composes better than special-casing
   `"n/a"`.
7. **What about the `train-full-lmq-v2-lat` (v2 t10n) cache?** Survey shows it's only in
   eu-west4. Any old-t10n smoke / re-eval will be pinned to eu-west4 by `any` mode unless
   the caller explicitly mirrors first. This is fine — v2 is legacy and shouldn't drive
   future fires.
8. **Recipe presets × region-strategy**: a recipe like `--recipe bin5` could opt into a
   particular region-strategy. Decision: recipes set knobs but don't override
   `--region-strategy`; the caller's strategy choice (or default) always wins. Otherwise
   the `bin5` recipe could surprise the user by pinning to a stale region.

## Open question: would `tomat train` ever *want* to register the picked region with the .sh / authors?

For the .sh migration (task #341 / sibling spec), the author writes:

```bash
tomat train --resume --parent bin5-fs-tpu --from-step 80000 --target-steps 82000 \
  --tpu v6e-16 --recipe bin5
```

…and the CLI handles region selection internally. The author never types `--zone us-east5-b`
again. Recipe-preset thinking: a "recipe" might want to express `region_strategy: explicit,
region: us-east5-a` for hypothesis-testing scripts, but the default (no `--region-strategy`)
should be `auto` for resumes / `any` for from-scratch.

## Cross-references

- Spec 61 — fire records + parent invariants. This spec extends 61's manifest shape.
- Task #99 — zone-local cache via runtime GCE-metadata zone detect (done).
- Task #210 — trainer writes ckpts to local-region bucket (done).
- Task #268 — iris multi-pool cascade-fallback (Will's region-lock observation). Worth
  reading before Phase A.
- Task #341 — migrate `scripts/fires/*.sh` to `tomat train`. Region strategy is one of the
  knobs the migration absorbs.
- Memory: [[cross-region-codec-mirror]], [[cross-region-eval-egress]], [[iris-zephyr-cache-race]].
