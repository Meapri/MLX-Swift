import mlx.core as mx
import pytest

import mlx_vlm.models.rope_utils as rope_utils
from mlx_vlm.models.rope_utils import (
    MRoPERotaryEmbedding,
    apply_multimodal_rotary_pos_emb,
    apply_rotary_pos_emb_even_odd,
)


def _max_diff(a, b):
    return mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item()


def _assert_pair_close(actual, expected, *, atol=1e-4):
    mx.eval(*actual, *expected)
    assert _max_diff(actual[0], expected[0]) < atol
    assert _max_diff(actual[1], expected[1]) < atol


def _disable_metal_fast_path(fn):
    has_metal = rope_utils._HAS_METAL
    rope_utils._HAS_METAL = False
    try:
        return fn()
    finally:
        rope_utils._HAS_METAL = has_metal


def _position_ids(batch=2, seq_len=4):
    base = mx.arange(batch * seq_len, dtype=mx.int32).reshape(batch, seq_len)
    return mx.stack([base, base + 3, base + 7])


@pytest.mark.parametrize(
    "style",
    [
        "chunked",
        "interleaved",
        "sectioned_half_split",
        "sectioned_even_odd",
        "split_select",
    ],
)
def test_mrope_apply_rotary_fast_path_matches_fallback(style):
    mx.random.seed(0)
    q = mx.random.normal((2, 3, 4, 10)).astype(mx.float32)
    k = mx.random.normal((2, 2, 4, 10)).astype(mx.float32)
    position_ids = _position_ids()
    kwargs = {
        "dim": 8,
        "base": 10000,
        "mrope_section": [2, 1, 1],
        "style": style,
    }

    rotary = MRoPERotaryEmbedding(**kwargs)
    fast = rotary.apply_rotary(q, k, position_ids)

    fallback = MRoPERotaryEmbedding(**kwargs)
    fallback.fused_apply = False
    expected = fallback.apply_rotary(q, k, position_ids)

    _assert_pair_close(fast, expected)
    _assert_pair_close((fast[0][..., 8:], fast[1][..., 8:]), (q[..., 8:], k[..., 8:]))


@pytest.mark.parametrize("style", ["sectioned_half_split", "sectioned_even_odd"])
def test_sectioned_precomputed_rotary_fast_path_matches_fallback(style):
    mx.random.seed(1)
    q = mx.random.normal((2, 3, 4, 10)).astype(mx.float32)
    k = mx.random.normal((2, 2, 4, 10)).astype(mx.float32)
    cos = mx.random.normal((3, 2, 4, 8)).astype(mx.float32)
    sin = mx.random.normal((3, 2, 4, 8)).astype(mx.float32)
    kwargs = {"mrope_section": [2, 1, 1], "style": style}

    fast = apply_multimodal_rotary_pos_emb(q, k, cos, sin, **kwargs)
    expected = _disable_metal_fast_path(
        lambda: apply_multimodal_rotary_pos_emb(q, k, cos, sin, **kwargs)
    )

    _assert_pair_close(fast, expected)
    _assert_pair_close((fast[0][..., 8:], fast[1][..., 8:]), (q[..., 8:], k[..., 8:]))


@pytest.mark.parametrize("cos_layout", ["half", "full"])
def test_even_odd_precomputed_rotary_fast_path_matches_fallback(cos_layout):
    mx.random.seed(2)
    q = mx.random.normal((2, 3, 4, 10)).astype(mx.float32)
    k = mx.random.normal((2, 2, 4, 10)).astype(mx.float32)
    cos = mx.random.normal((2, 4, 8)).astype(mx.float32)
    sin = mx.random.normal((2, 4, 8)).astype(mx.float32)

    fast = apply_rotary_pos_emb_even_odd(q, k, cos, sin, cos_layout=cos_layout)
    expected = _disable_metal_fast_path(
        lambda: apply_rotary_pos_emb_even_odd(q, k, cos, sin, cos_layout=cos_layout)
    )

    _assert_pair_close(fast, expected)
    _assert_pair_close((fast[0][..., 8:], fast[1][..., 8:]), (q[..., 8:], k[..., 8:]))
