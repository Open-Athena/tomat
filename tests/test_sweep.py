"""Tests for the training-sweep config helper."""

from tomat.training.sweep import (
    CODEC_TOKENS_PER_VALUE,
    SweepConfig,
    all_configs,
    valid_configs,
)


def test_tokens_per_value_match_codec_semantics():
    assert CODEC_TOKENS_PER_VALUE["tomol_3byte"] == 3
    assert CODEC_TOKENS_PER_VALUE["two_token_9_12"] == 2
    assert CODEC_TOKENS_PER_VALUE["fp16_1token"] == 1


def test_density_token_count_is_P_cubed_times_stride():
    for codec, stride in CODEC_TOKENS_PER_VALUE.items():
        for p in (8, 12, 14, 16, 20):
            cfg = SweepConfig(codec=codec, patch_size=p)
            assert cfg.density_tokens == p ** 3 * stride


def test_sweep_label_is_unique():
    labels = [c.label for c in all_configs()]
    assert len(labels) == len(set(labels))


def test_valid_configs_at_8k_drops_overflow():
    valid = valid_configs(context_budget=8192)
    valid_labels = {c.label for c in valid}
    # These should fit:
    assert "two_token_9_12-P14" in valid_labels
    assert "fp16_1token-P16" in valid_labels
    assert "tomol_3byte-P12" in valid_labels
    # These overflow 8k:
    assert "tomol_3byte-P14" not in valid_labels
    assert "tomol_3byte-P16" not in valid_labels
    assert "two_token_9_12-P16" not in valid_labels


def test_total_vocab_size_monotonic_in_codec():
    cfgs = {c.codec: c.total_vocab_size for c in all_configs() if c.patch_size == 14}
    # 3-byte (1024 density vocab) < 9+12 (4608) < fp16 (65536)
    assert cfgs["tomol_3byte"] < cfgs["two_token_9_12"] < cfgs["fp16_1token"]
    # Non-density vocab = 18 + 118 + 1024 + 1024 = 2184
    assert cfgs["tomol_3byte"] == 2184 + 1024
    assert cfgs["fp16_1token"] == 2184 + 65_536
