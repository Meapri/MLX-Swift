import mlx.core as mx
import pytest

import mlx_vlm.models.rope_utils as rope_utils
from mlx_vlm.models.rope_utils import (
    MRoPERotaryEmbedding,
    apply_multimodal_rotary_pos_emb,
    apply_mrope_frequency_layout,
    apply_rotary_pos_emb_even_odd,
    compute_mrope_frequencies,
    compute_selected_mrope_cos_sin,
    mrope_section_selectors,
    mrope_position_selector,
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


def _reference_mrope_frequency_layout(freqs, mrope_section, style):
    if style == "chunked":
        split_indices = mx.cumsum(mx.array(mrope_section, dtype=mx.int32))[:-1]
        return mx.concatenate(
            [
                chunk[axis]
                for axis, chunk in enumerate(
                    mx.split(freqs, split_indices, axis=-1)
                )
            ],
            axis=-1,
        )
    if style == "interleaved":
        selected = []
        for idx in range(sum(mrope_section)):
            axis = 0
            if idx % 3 == 1 and idx < mrope_section[1] * 3:
                axis = 1
            elif idx % 3 == 2 and idx < mrope_section[2] * 3:
                axis = 2
            selected.append(freqs[axis, ..., idx : idx + 1])
        return mx.concatenate(selected, axis=-1)
    if style == "split_select":
        split_indices = mx.cumsum(mx.array(mrope_section, dtype=mx.int32))[:-1]
        return mx.concatenate(
            [
                chunk[axis % 3]
                for axis, chunk in enumerate(
                    mx.split(freqs, split_indices, axis=-1)
                )
            ],
            axis=-1,
        )
    return freqs


@pytest.mark.parametrize("style", ["chunked", "interleaved", "split_select"])
def test_selected_frequency_fast_path_matches_layout_reference(style):
    mx.random.seed(3)
    position_ids = _position_ids(batch=2, seq_len=4)
    mrope_section = [2, 2, 2]
    inv_freq = mx.random.normal((sum(mrope_section),)).astype(mx.float32)
    position_selector = mrope_position_selector(
        style,
        mrope_section,
        inv_freq.shape[0],
    )

    fast = compute_mrope_frequencies(
        position_ids,
        inv_freq,
        mrope_section,
        style=style,
        position_selector=position_selector,
    )
    freqs = position_ids.astype(mx.float32)[..., None] * inv_freq
    layout = apply_mrope_frequency_layout(
        freqs,
        mrope_section,
        style=style,
    )
    expected = _reference_mrope_frequency_layout(freqs, mrope_section, style)

    mx.eval(fast, layout, expected)
    assert _max_diff(fast, expected) < 1e-4
    assert _max_diff(layout, expected) < 1e-4


def _reference_section_selected_mrope_cos_sin(
    position_ids,
    inv_freq,
    mrope_section,
    position_axes,
    *,
    interleave_sections=(),
):
    starts = [0]
    for dim in mrope_section[:-1]:
        starts.append(starts[-1] + dim)

    section_freqs = []
    for start, dim, position_axis in zip(starts, mrope_section, position_axes):
        section_positions = position_ids[position_axis].astype(mx.float32)[..., None]
        section_freqs.append(section_positions * inv_freq[start : start + dim])

    parts = []
    interleave_sections = tuple(interleave_sections)
    for section_idx, section_freq in enumerate(section_freqs):
        if section_idx not in interleave_sections:
            parts.append(section_freq)
            continue
        if section_idx != interleave_sections[0]:
            continue
        interleaved = []
        for offset in range(section_freq.shape[-1]):
            for interleave_idx in interleave_sections:
                interleaved.append(section_freqs[interleave_idx][..., offset])
        parts.append(mx.stack(interleaved, axis=-1))

    freqs = mx.concatenate(parts, axis=-1)
    emb = mx.repeat(freqs, repeats=2, axis=-1)
    return mx.cos(emb), mx.sin(emb)


def test_mrope_section_selectors_interleave_selected_sections():
    position_selector, frequency_selector = mrope_section_selectors(
        [2, 2, 2],
        position_axes=(1, 2, 0),
        interleave_sections=(0, 1),
    )

    assert position_selector.tolist() == [1, 2, 1, 2, 0, 0]
    assert frequency_selector.tolist() == [0, 2, 1, 3, 4, 5]


def test_selected_mrope_cos_sin_matches_reference():
    mx.random.seed(4)
    mrope_section = [2, 2, 2]
    position_axes = (1, 2, 0)
    interleave_sections = (0, 1)
    position_ids = _position_ids(batch=2, seq_len=4)
    inv_freq = mx.random.normal((sum(mrope_section),)).astype(mx.float32)
    position_selector, frequency_selector = mrope_section_selectors(
        mrope_section,
        position_axes=position_axes,
        interleave_sections=interleave_sections,
    )

    fast = compute_selected_mrope_cos_sin(
        position_ids,
        inv_freq,
        position_selector,
        frequency_selector,
    )
    expected = _reference_section_selected_mrope_cos_sin(
        position_ids,
        inv_freq,
        mrope_section,
        position_axes,
        interleave_sections=interleave_sections,
    )

    _assert_pair_close(fast, expected)
