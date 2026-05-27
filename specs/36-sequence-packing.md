# Sequence packing — concat N patches per row, block-diagonal attention

**Date:** 2026-05-27
**Status:** WIP — implementing scheme 2a per user direction.
**Related:** spec 18 (LMQ codec), spec 28 (MaskGIT), spec 30 (init-cls bug post-mortem), the v3-P14 tokenization config.

## Motivation

At our current most-aggressive tokenization (`v3` cube, `P=14`, LMQ, single patch per row), one row's `input_ids` looks like:

```
[preamble_v3 | DENS_START | density_tokens] [PAD] [PAD] ... [PAD]
└── ~210 tokens               └── 2744       ─────── ~5000 PAD ───────
                                 (P³ density tokens)
                                                               ↑
                                                  total row width = pad_to (e.g. 8192)
```

The density-token block dominates the loss signal but everything is paid for compute-wise — TPU does flops over the whole `Pos × Vocab` softmax/projection. At `pad_to=8192`, **65–70% of every row is PAD**, contributing zero gradient. v3-P14 with `pad_to=4608` is better (~40% PAD) but still wasteful, and v2-P14 / future P>14 configs swing back to >50% PAD.

If we *concatenate* N independent patches per row instead of padding, we burn the same per-row TPU compute on N× more loss-bearing tokens. For `P=14` at `pad_to=8192`, N=2 fits comfortably (2 × ~3000 ≈ 6000 tokens, 2192 PAD remainder); for `P<14` or larger `pad_to` we get N=3+.

## Layout (scheme 2a)

Each packed row is:

```
[seq_1] [seq_2] ... [seq_N] [PAD ...]
```

where each `seq_k = [preamble_k | DENS_START | density_tokens_k]`. The sub-sequences are *whole* patches — no slicing, no document overlap. `N` is chosen at tokenization time, sized to `pad_to // (p99(preamble_len) + density_tokens_per_patch)` so that with overwhelming probability we never overflow a row (oversized preambles fall through the existing pad-overflow skip path, same as today).

**Sentinel choice.** Each `seq_k` already begins with deterministic preamble tokens (e.g. `[LATTICE_START]` in v2 or `[ATOMS_START]` in v3). Using one of those as the segment-boundary sentinel is unreliable: their position drifts when the preamble shape changes. Cleanest approach: emit `[PAD]` (token id 0) as a single-token separator at the start of every `seq_k` *except seq_1*, and rely on Levanter's existing `block_cross_document_attention` machinery, which computes `segment_ids = cumsum(tokens == eos_id)` and wires that into the attention mask. We treat `[PAD]` as our EOS-equivalent — it's already the trailing-pad token, so post-`seq_N` PAD bytes naturally land in their own (ignored) trailing segment.

Actually, even cleaner: **leave the existing trailing PAD as PAD (token 0), and emit a single `[PAD]` between adjacent sub-sequences**. The cumsum-segment computation then gives:

```
tokens     : [...seq_1...] [PAD] [...seq_2...] [PAD] [...seq_3...] [PAD] [PAD] [PAD] ...
seg_id     :  0  0  0  0    1    1  1  1  1    2    2  2  2  2    3    3   3   3   ...
```

— each sub-sequence and the trailing PAD region each get a distinct segment. Cross-segment attention is blocked. Within each segment, causal masking already restricts attention to prior positions of the same sub-sequence.

This matches scheme 2a exactly: block-diagonal causal attention, no cross-boundary leakage, PAD remainder isolated.

## Loss masking

The trainer's per-position loss already zeros out PAD-target positions via the standard `loss_weight = causal_mask & (next_token != ignore_id)` path (Levanter's `GrugLmExample.causal` does this when `eos_id` is set). We extend that here in two ways:

1. **Preamble-loss policy reuse.** The existing `TOMAT_DENSITY_ONLY_LOSS=1` knob already zeros CE everywhere except density-target positions. Under packed rows this still does the right thing — preamble tokens of `seq_2..N` are graded by exactly the same rule as `seq_1`'s preamble.
2. **Boundary-token policy.** The single-`[PAD]` separator's *next* token is the next sub-sequence's first preamble token. With `density_only=True` this is already zero-loss; with `density_only=False` (CE on preamble) it still scores a categorical loss for the boundary→preamble transition — slightly wrong but small (one position per sub-sequence). For the first phase we accept this; if eval shows it matters, we'll zero loss on positions immediately following a segment-boundary `[PAD]`.

## Tokenizer changes (`scripts/tokenize_patches.py`)

Add a `--pack` flag (default off for back-compat). When set:

1. For each material, generate the existing list of patch-token lists (same `make_sample` + `tokenize` flow).
2. Greedily concatenate patches into a row buffer: append a `[PAD]` separator (token id 0) between adjacent patches; stop adding when the next patch wouldn't fit in `pad_to`. Pad remainder with `[PAD]`.
3. Each output row's `input_ids` is one packed row; one material still produces ~`M/N` rows (so `patches_per_material` is no longer the row count — track the *patch* count separately in `meta.json` for downstream eval).
4. `meta.json` gains: `packed: true`, `patches_per_row_p50/p99`, total `n_patches`.

A new column `boundary_positions: list[int32]` (positions of every `seq_k` start, k>=1) is **not** needed since the trainer derives segments from the boundary-PAD sentinel at row-load time. Skipping the sidecar keeps parquet schema compatible with the unpacked case.

## Trainer changes (`marin/train_tomat_tpu.py` + `marin/qwen3_density.py`)

The `LmDataConfig` already has a `block_cross_document_attention: bool = True` knob, and `PrebuiltLmDataset` (the dataset format we use) forwards `eos_id` + `block_cross_document_attention` into `GrugLmExample.causal`, which converts `cumsum(tokens == eos_id)` into segment IDs and wires the segment mask into attention. So at the data-side, the changes are:

- `LmDataConfig(..., block_cross_document_attention=True)` — already the default; just don't override it.
- Plumb `eos_id=PAD_ID(=0)` into the `LmDataConfig`. Today our `tokenizer="passthrough"` flow doesn't set `eos_token_id` (the JSON log shows `enforce_eos: true` but with no real EOS, no segments get created). We need to set `eos_token_id` on the `PassthroughTokenizer` or pass it explicitly through `LmDataConfig`. **TODO during implementation: verify the cleanest path — set it on `PassthroughTokenizer.__init__`, or override `the_tokenizer.eos_token_id` post-build, or add an env var.**

On the loss side:

- `Qwen3DensityLMHeadModel.compute_next_token_loss`: existing code already passes `example.attn_mask` into `self.activations(...)`. Because `attn_mask` arrives with `segment_ids` set, the causal-attention layer already applies the block-diagonal mask. No changes needed in the activations pass.
- `density_aware_loss`: rolls the input by -1 to get next-token targets. At sub-sequence boundaries this rolls the **last density token of `seq_k`** to predict the **boundary `[PAD]`**, which is wrong-target — the model would be graded as if it should predict PAD after density. Fix: zero the loss weight at boundary positions (where `targets == PAD_ID`). The existing `roll(-1)` + `not_last` mask handles the tail; we add `is_pad_target = targets == PAD_ID` and AND it into the loss weight. (Already happens for the trailing PAD tail, but not for the inline `[PAD]` separators).
- `maskgit_aware_loss`: targets are pre-mask original tokens (no roll). The mask schedule never picks `[PAD]` positions (they're not density tokens). So no change needed — the boundary `[PAD]` just sits there.
- SS path (`Qwen3SSLMHeadModel`): same as density — only density-target inputs are eligible for mixing, so boundary `[PAD]` is never touched.

The `mg`/`ss` paths build their own `LmExample`s mid-forward (`AttentionMask(is_causal=False)` for MaskGIT). For MaskGIT we must propagate the `segment_ids` from the incoming `example.attn_mask` into the new bidirectional mask, or the masked forward will leak across segments. The fix: `bidir_mask = AttentionMask(is_causal=False).with_segment_ids(...)` using the incoming segment IDs.

## Backward compatibility

The `--pack` flag defaults to off. Existing v3 datasets (un-packed) read as before:

- `block_cross_document_attention=True` is harmless when there are zero `[PAD]` sentinels mid-row — `cumsum(...)==0` everywhere up to the trailing PAD tail, all one segment.
- The trainer's loss code's added `targets != PAD_ID` mask is also harmless on un-packed data (existing rows had no inline PADs).

To make the trainer assume packed data for newly-tokenized packed shards, we read `meta.json`'s `packed: bool` field and set the trainer's `eos_id` only when packing is on. Un-packed runs continue with `eos_id=None` (no segmentation).

`TOMAT_PACKED=1` env can force-enable the packed code path for ad-hoc smokes.

## Smoke test plan

1. Build a tiny synthetic dataset (1 material, 4 patches) tokenized with `--pack` → one row with N=2 packed sub-sequences + trailing PAD. Inspect the rows by hand.
2. Build a model on CPU with `vocab_size = max(tokens) + 1`, run a single forward pass on the packed row.
3. Run the same forward on `seq_1` alone (right-padded). Assert the logits at every position of `seq_1` in the packed row match the standalone forward within fp32 tolerance — verifies no cross-boundary attention.
4. Same check for `seq_2` (slice it out, right-pad to row width, forward, compare positions).
5. Loss check: run `compute_next_token_loss` with `density_only=True`; assert the loss is finite and only density positions contribute (loss-weight non-zero).

This lives in `tests/test_packed_attention.py`.

## Open questions / decisions

- **Where to set `eos_id` for `passthrough` tokenizer.** Smallest patch: monkey-patch `PassthroughTokenizer.__init__` from `train_tomat_tpu.py` to expose `eos_token_id=PAD_ID` whenever `TOMAT_PACKED=1`. Cleaner option: a `LmDataConfig.eos_id_override` field in Levanter — leave for upstream.
- **Shuffling within a packed row.** Block-shuffle today operates at the row level (`BlockShuffleConfig(io_block_size=M=32)` so one block = one material). Packed rows mix N patches *from the same material* deterministically (same anchor draw order); cross-material mixing still happens via row-shuffling. We accept the within-row co-correlation (better than today's "all 32 patches from one material in a single batch position when bs<M") and revisit if grad-noise plots suggest it matters.
- **Do we need `[DOC_BOUNDARY]` as a distinct token?** No. `[PAD]` is reused as the segment-boundary sentinel; the trailing PAD region is already zero-loss for density loss, and CE on a "predict PAD next" target is the same as predicting any other-vocab token from preamble context — small constant overhead, ignorable.
- **MaskGIT with packing.** MaskGIT mode requires propagating `segment_ids` through the model's internally-constructed bidirectional `attn_mask`. Validated by the smoke (item 5 above).
- **Loss-weight bookkeeping under `TOMAT_DENSITY_ONLY_LOSS=0`.** Existing CE path scores every non-PAD position; the inline `[PAD]` separator is a single PAD target per boundary, ~1/3000 of total loss tokens at N=2 — small but nonzero bias. The trainer's added `targets != PAD_ID` mask removes this for the AR-density path. Leave the MaskGIT path's per-position CE as-is for now (MaskGIT only grades masked density positions anyway).

## Estimated impact

For v3-P14 today (`P=14`, density 2744 tokens, preamble ~210 tokens, `pad_to=4608`):

- Un-packed: 210 + 1 + 2744 = 2955 effective tokens / 4608 row = **64% utilization**.
- N=1: same as today.

For v3-P14 at `pad_to=8192` (our previous standard):

- Un-packed: 2955 / 8192 = **36% utilization**.
- Packed N=2: 2 × 2955 + 1 PAD-sep = 5911 / 8192 = **72% utilization**, ~2× effective batch size at fixed wall-clock.

For 200M v6e-16 at 13% MFU today, doubling effective throughput would imply ~26% MFU if the matmul → memory ratio doesn't shift (it should improve slightly: more tokens per HBM-resident weight load).
