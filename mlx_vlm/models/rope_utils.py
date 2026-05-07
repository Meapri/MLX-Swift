from functools import lru_cache
from typing import Optional, Sequence

import mlx.core as mx

_HAS_METAL = mx.metal.is_available()
_HALF_SPLIT = "half_split"
_EVEN_ODD = "even_odd"
_HALF_COS = "half"
_FULL_COS = "full"
_POSITION_SELECTOR_STYLES = {
    "chunked",
    "interleaved",
    "sectioned_half_split",
    "sectioned_even_odd",
    "split_select",
}
_SELECTED_FREQUENCY_STYLES = {"chunked", "interleaved", "split_select"}


def _cumulative_splits(lengths: Sequence[int]):
    return mx.cumsum(mx.array(lengths, dtype=mx.int32))[:-1]


def _interleaved_position_selector(mrope_section: Sequence[int], freq_dim: int):
    selector = [0] * freq_dim
    for dim, offset in enumerate((1, 2), start=1):
        for idx in range(offset, min(mrope_section[dim] * 3, freq_dim), 3):
            selector[idx] = dim
    return mx.array(selector, dtype=mx.int32)


def _chunked_position_selector(mrope_section: Sequence[int], freq_dim: int):
    selector = [0] * freq_dim
    offset = mrope_section[0]
    for dim, length in enumerate(mrope_section[1:], start=1):
        for idx in range(offset, min(offset + length, freq_dim)):
            selector[idx] = dim
        offset += length
    return mx.array(selector, dtype=mx.int32)


@mx.compile
def _selected_mrope_freqs(position_ids, inv_freq, position_selector):
    positions = mx.take(position_ids, position_selector, axis=0).transpose(1, 2, 0)
    return positions.astype(mx.float32) * inv_freq


def _pairing_for_style(style: str):
    if style in {"sectioned_even_odd", "split_select", "ernie_3d"}:
        return _EVEN_ODD
    return _HALF_SPLIT


def mrope_position_selector(style: str, mrope_section: Sequence[int], freq_dim: int):
    if style == "interleaved":
        return _interleaved_position_selector(mrope_section, freq_dim)
    return _chunked_position_selector(mrope_section, freq_dim)


@lru_cache(maxsize=None)
def _mrope_apply_kernel(rotary_dim: int, position_ndim: int, pairing: str):
    if not _HAS_METAL:
        return None

    if position_ndim == 2:
        position_expr = "position_ids[b * q_len + t]"
    else:
        position_expr = "position_ids[(axis * q_bsz + b) * q_len + t]"

    if pairing == _EVEN_ODD:
        pair_source = f"""
        if (d >= {rotary_dim}) {{
            if (is_q) {{
                q_out[local] = q[local];
            }} else {{
                k_out[local] = k[local];
            }}
            return;
        }}

        if ((d & 1) != 0) {{
            return;
        }}

        int freq_idx = d / 2;
        int pair_d = d + 1;
        int axis = int(position_selector[freq_idx]);
        float pos = static_cast<float>({position_expr});
        float angle = pos * static_cast<float>(inv_freq[freq_idx]);
        float c = metal::cos(angle);
        float s = metal::sin(angle);
        """
    else:
        pair_source = f"""
        if (d >= {rotary_dim}) {{
            if (is_q) {{
                q_out[local] = q[local];
            }} else {{
                k_out[local] = k[local];
            }}
            return;
        }}

        if (d >= half_dim) {{
            return;
        }}

        int freq_idx = d;
        int pair_d = d + half_dim;
        int axis = int(position_selector[freq_idx]);
        float pos = static_cast<float>({position_expr});
        float angle = pos * static_cast<float>(inv_freq[freq_idx]);
        float c = metal::cos(angle);
        float s = metal::sin(angle);
        """

    source = f"""
        uint elem = thread_position_in_grid.x;

        const int q_bsz = q_shape[0];
        const int q_heads = q_shape[1];
        const int q_len = q_shape[2];
        const int q_dim = q_shape[3];
        const int k_heads = k_shape[1];
        const int k_dim = k_shape[3];
        const int half_dim = {rotary_dim // 2};
        const int q_size = q_bsz * q_heads * q_len * q_dim;
        const int k_size = q_bsz * k_heads * q_len * k_dim;

        if (elem >= uint(q_size + k_size)) {{
            return;
        }}

        bool is_q = elem < uint(q_size);
        int local = is_q ? int(elem) : int(elem) - q_size;
        int D = is_q ? q_dim : k_dim;
        int H = is_q ? q_heads : k_heads;
        int d = local % D;
        int tmp = local / D;
        int t = tmp % q_len;
        tmp = tmp / q_len;
        int h = tmp % H;
        int b = tmp / H;
        int base = ((b * H + h) * q_len + t) * D;

        {pair_source}

        if (is_q) {{
            float x = static_cast<float>(q[local]);
            float xp = static_cast<float>(q[base + pair_d]);
            q_out[local] = static_cast<T>(x * c - xp * s);
            q_out[base + pair_d] = static_cast<T>(xp * c + x * s);
        }} else {{
            float x = static_cast<float>(k[local]);
            float xp = static_cast<float>(k[base + pair_d]);
            k_out[local] = static_cast<T>(x * c - xp * s);
            k_out[base + pair_d] = static_cast<T>(xp * c + x * s);
        }}
    """

    return mx.fast.metal_kernel(
        name=f"mrope_apply_{pairing}_{rotary_dim}_{position_ndim}d",
        input_names=["q", "k", "position_ids", "inv_freq", "position_selector"],
        output_names=["q_out", "k_out"],
        source=source,
    )


def _fast_mrope_apply(
    kernel,
    q,
    k,
    position_ids,
    inv_freq,
    position_selector,
):
    q_size = q.size
    k_size = k.size
    outputs = kernel(
        inputs=[q, k, position_ids, inv_freq, position_selector],
        template=[("T", q.dtype)],
        grid=(q_size + k_size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[q.shape, k.shape],
        output_dtypes=[q.dtype, k.dtype],
    )
    return outputs[0], outputs[1]


@lru_cache(maxsize=None)
def _compiled_mrope_apply(rotary_dim: int, position_ndim: int, pairing: str):
    kernel = _mrope_apply_kernel(rotary_dim, position_ndim, pairing)
    if kernel is None:
        return None

    @mx.compile
    def apply(q, k, position_ids, inv_freq, position_selector):
        return _fast_mrope_apply(
            kernel,
            q,
            k,
            position_ids,
            inv_freq,
            position_selector,
        )

    return apply


@lru_cache(maxsize=None)
def _rotary_apply_kernel(
    rotary_dim: int,
    pairing: str,
    cos_ndim: int,
    sectioned: bool,
    cos_layout: str,
):
    if not _HAS_METAL:
        return None

    if pairing == _EVEN_ODD:
        pair_source = f"""
        if (d >= {rotary_dim}) {{
            if (is_q) {{
                q_out[local] = q[local];
            }} else {{
                k_out[local] = k[local];
            }}
            return;
        }}

        if ((d & 1) != 0) {{
            return;
        }}

        int freq_idx = d / 2;
        int pair_d = d + 1;
        """
    else:
        pair_source = f"""
        if (d >= {rotary_dim}) {{
            if (is_q) {{
                q_out[local] = q[local];
            }} else {{
                k_out[local] = k[local];
            }}
            return;
        }}

        if (d >= half_dim) {{
            return;
        }}

        int freq_idx = d;
        int pair_d = d + half_dim;
        """

    if pairing == _HALF_SPLIT:
        cos_freq_idx = "freq_idx"
        pair_cos_freq_idx = "pair_d"
    elif cos_layout == _FULL_COS:
        cos_freq_idx = "d"
        pair_cos_freq_idx = "pair_d"
    else:
        cos_freq_idx = "freq_idx"
        pair_cos_freq_idx = "freq_idx"

    if sectioned:
        cos_expr = (
            f"cos[((axis * q_bsz + b) * q_len + t) * {rotary_dim} + "
            f"{cos_freq_idx}]"
        )
        sin_expr = (
            f"sin[((axis * q_bsz + b) * q_len + t) * {rotary_dim} + "
            f"{cos_freq_idx}]"
        )
        pair_cos_expr = (
            f"cos[((axis * q_bsz + b) * q_len + t) * {rotary_dim} + "
            f"{pair_cos_freq_idx}]"
        )
        pair_sin_expr = (
            f"sin[((axis * q_bsz + b) * q_len + t) * {rotary_dim} + "
            f"{pair_cos_freq_idx}]"
        )
        selector_source = "int axis = int(position_selector[freq_idx]);"
        input_names = ["q", "k", "cos", "sin", "position_selector"]
    elif cos_ndim == 4:
        cos_expr = (
            f"cos[((0 * q_bsz + b) * q_len + t) * {rotary_dim} + {cos_freq_idx}]"
        )
        sin_expr = (
            f"sin[((0 * q_bsz + b) * q_len + t) * {rotary_dim} + {cos_freq_idx}]"
        )
        pair_cos_expr = (
            f"cos[((0 * q_bsz + b) * q_len + t) * {rotary_dim} + "
            f"{pair_cos_freq_idx}]"
        )
        pair_sin_expr = (
            f"sin[((0 * q_bsz + b) * q_len + t) * {rotary_dim} + "
            f"{pair_cos_freq_idx}]"
        )
        selector_source = ""
        input_names = ["q", "k", "cos", "sin"]
    else:
        cos_expr = f"cos[(b * q_len + t) * {rotary_dim} + {cos_freq_idx}]"
        sin_expr = f"sin[(b * q_len + t) * {rotary_dim} + {cos_freq_idx}]"
        pair_cos_expr = f"cos[(b * q_len + t) * {rotary_dim} + {pair_cos_freq_idx}]"
        pair_sin_expr = f"sin[(b * q_len + t) * {rotary_dim} + {pair_cos_freq_idx}]"
        selector_source = ""
        input_names = ["q", "k", "cos", "sin"]

    source = f"""
        uint elem = thread_position_in_grid.x;

        const int q_bsz = q_shape[0];
        const int q_heads = q_shape[1];
        const int q_len = q_shape[2];
        const int q_dim = q_shape[3];
        const int k_heads = k_shape[1];
        const int k_dim = k_shape[3];
        const int half_dim = {rotary_dim // 2};
        const int q_size = q_bsz * q_heads * q_len * q_dim;
        const int k_size = q_bsz * k_heads * q_len * k_dim;

        if (elem >= uint(q_size + k_size)) {{
            return;
        }}

        bool is_q = elem < uint(q_size);
        int local = is_q ? int(elem) : int(elem) - q_size;
        int D = is_q ? q_dim : k_dim;
        int H = is_q ? q_heads : k_heads;
        int d = local % D;
        int tmp = local / D;
        int t = tmp % q_len;
        tmp = tmp / q_len;
        int h = tmp % H;
        int b = tmp / H;
        int base = ((b * H + h) * q_len + t) * D;

        {pair_source}
        {selector_source}
        float c = static_cast<float>({cos_expr});
        float s = static_cast<float>({sin_expr});
        float cp = static_cast<float>({pair_cos_expr});
        float sp = static_cast<float>({pair_sin_expr});

        if (is_q) {{
            float x = static_cast<float>(q[local]);
            float xp = static_cast<float>(q[base + pair_d]);
            q_out[local] = static_cast<T>(x * c - xp * s);
            q_out[base + pair_d] = static_cast<T>(xp * cp + x * sp);
        }} else {{
            float x = static_cast<float>(k[local]);
            float xp = static_cast<float>(k[base + pair_d]);
            k_out[local] = static_cast<T>(x * c - xp * s);
            k_out[base + pair_d] = static_cast<T>(xp * cp + x * sp);
        }}
    """

    section_suffix = "sectioned" if sectioned else "preselected"
    return mx.fast.metal_kernel(
        name=(
            f"rotary_apply_{pairing}_{cos_layout}_{section_suffix}_"
            f"{rotary_dim}_{cos_ndim}d"
        ),
        input_names=input_names,
        output_names=["q_out", "k_out"],
        source=source,
    )


def _fast_rotary_apply(kernel, q, k, cos, sin, position_selector=None):
    inputs = [q, k, cos, sin]
    if position_selector is not None:
        inputs.append(position_selector)
    outputs = kernel(
        inputs=inputs,
        template=[("T", q.dtype)],
        grid=(q.size + k.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[q.shape, k.shape],
        output_dtypes=[q.dtype, k.dtype],
    )
    return outputs[0], outputs[1]


@lru_cache(maxsize=None)
def _compiled_rotary_apply(
    rotary_dim: int,
    pairing: str,
    cos_ndim: int,
    sectioned: bool,
    cos_layout: str,
):
    kernel = _rotary_apply_kernel(
        rotary_dim,
        pairing,
        cos_ndim,
        sectioned,
        cos_layout,
    )
    if kernel is None:
        return None

    @mx.compile
    def apply(q, k, cos, sin, position_selector):
        return _fast_rotary_apply(kernel, q, k, cos, sin, position_selector)

    return apply


def get_mrope_section(
    *,
    rope_scaling: Optional[dict] = None,
    rope_parameters: Optional[dict] = None,
    default: Sequence[int] = (24, 20, 20),
):
    rope_scaling = rope_scaling or {}
    rope_parameters = rope_parameters or {}
    return list(
        rope_parameters.get("mrope_section")
        or rope_scaling.get("mrope_section")
        or default
    )


def compute_inv_freq(dim: int, base: float):
    return 1.0 / (base ** (mx.arange(0, dim, 2).astype(mx.float32) / dim))


@mx.compile
def _apply_selected_mrope_frequency_layout(freqs, position_selector):
    indices = mx.broadcast_to(
        position_selector[None, None, None, :],
        (1, freqs.shape[1], freqs.shape[2], freqs.shape[3]),
    )
    return mx.take_along_axis(freqs, indices, axis=0)[0]


def mrope_section_selectors(
    mrope_section: Sequence[int],
    position_axes: Sequence[int],
    *,
    interleave_sections: Sequence[int] = (),
):
    mrope_section = list(mrope_section)
    position_axes = list(position_axes)
    interleave_sections = list(interleave_sections)
    if len(position_axes) != len(mrope_section):
        raise ValueError("position_axes must match mrope_section length")

    offsets = [0]
    for length in mrope_section[:-1]:
        offsets.append(offsets[-1] + length)

    position_selector = []
    frequency_selector = []

    interleaved = set(interleave_sections)
    if interleave_sections:
        interleave_length = mrope_section[interleave_sections[0]]
        if any(mrope_section[idx] != interleave_length for idx in interleave_sections):
            raise ValueError("interleaved MRoPE sections must have equal length")
        for local_idx in range(interleave_length):
            for section_idx in interleave_sections:
                position_selector.append(position_axes[section_idx])
                frequency_selector.append(offsets[section_idx] + local_idx)

    for section_idx, length in enumerate(mrope_section):
        if section_idx in interleaved:
            continue
        offset = offsets[section_idx]
        for local_idx in range(length):
            position_selector.append(position_axes[section_idx])
            frequency_selector.append(offset + local_idx)

    return (
        mx.array(position_selector, dtype=mx.int32),
        mx.array(frequency_selector, dtype=mx.int32),
    )


@mx.compile
def compute_selected_mrope_cos_sin(
    position_ids,
    inv_freq,
    position_selector,
    frequency_selector,
):
    positions = mx.take(position_ids, position_selector, axis=0).transpose(1, 2, 0)
    freqs = positions.astype(mx.float32) * mx.take(inv_freq, frequency_selector)
    emb = mx.repeat(freqs, repeats=2, axis=-1)
    return mx.cos(emb), mx.sin(emb)


def apply_mrope_frequency_layout(
    freqs,
    mrope_section: Sequence[int],
    *,
    style: str = "interleaved",
):
    mrope_section = list(mrope_section)

    if style in _SELECTED_FREQUENCY_STYLES:
        position_selector = mrope_position_selector(
            style,
            mrope_section,
            freqs.shape[-1],
        )
        return _apply_selected_mrope_frequency_layout(freqs, position_selector)
    return freqs


def compute_mrope_frequencies(
    position_ids,
    inv_freq,
    mrope_section: Sequence[int],
    *,
    style: str = "interleaved",
    position_selector=None,
):
    if position_ids.ndim == 2:
        return position_ids.astype(mx.float32)[..., None] * inv_freq

    # Fast path
    if style in _SELECTED_FREQUENCY_STYLES:
        if position_selector is None:
            position_selector = mrope_position_selector(
                style,
                mrope_section,
                inv_freq.shape[0],
            )
        return _selected_mrope_freqs(position_ids, inv_freq, position_selector)

    # Slow path
    freqs = position_ids.astype(mx.float32)[..., None] * inv_freq
    return apply_mrope_frequency_layout(freqs, mrope_section, style=style)


class MRoPERotaryEmbedding:
    """Shared language-side rotary embedding for MRoPE models.

    ``style`` selects whether frequency layout is applied in the embedding
    itself (Qwen/GLM-OCR style) or deferred to Q/K application
    (PaddleOCR/GLM4V-style sectioned RoPE).
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000,
        *,
        rope_scaling: Optional[dict] = None,
        rope_parameters: Optional[dict] = None,
        mrope_section: Optional[Sequence[int]] = None,
        attention_scaling: float = 1.0,
        cast_output: bool = True,
        style: str = "interleaved",
    ):
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.style = style
        self.inv_freq = compute_inv_freq(dim, base)
        self.attention_scaling = attention_scaling
        self.cast_output = cast_output
        self.mrope_section = list(
            mrope_section
            if mrope_section is not None
            else get_mrope_section(
                rope_scaling=rope_scaling,
                rope_parameters=rope_parameters,
            )
        )
        if style in _POSITION_SELECTOR_STYLES:
            self.position_selector = mrope_position_selector(
                style,
                self.mrope_section,
                self.inv_freq.shape[0],
            )
        else:
            self.position_selector = None
        self.pairing = _pairing_for_style(style)
        self.fused_apply = self.position_selector is not None and _HAS_METAL
        self._compiled_apply = {} if self.fused_apply else None

    def __call__(self, x, position_ids):
        freqs = compute_mrope_frequencies(
            position_ids,
            self.inv_freq,
            self.mrope_section,
            style=self.style,
            position_selector=self.position_selector,
        )
        emb = mx.concatenate([freqs, freqs], axis=-1)
        cos = mx.cos(emb) * self.attention_scaling
        sin = mx.sin(emb) * self.attention_scaling

        if self.cast_output:
            return cos.astype(x.dtype), sin.astype(x.dtype)
        return cos, sin

    def apply_rotary(
        self,
        q,
        k,
        position_ids,
        *,
        unsqueeze_dim: int = 1,
        cast_output: bool = True,
    ):
        if (
            self.fused_apply
            and unsqueeze_dim == 1
            and position_ids.ndim in (2, 3)
            and q.ndim == 4
            and k.ndim == 4
        ):
            compiled_apply = self._compiled_apply.get(position_ids.ndim)
            if compiled_apply is None:
                compiled_apply = _compiled_mrope_apply(
                    self.dim, position_ids.ndim, self.pairing
                )
                if compiled_apply is not None:
                    self._compiled_apply[position_ids.ndim] = compiled_apply

            fast = None
            if compiled_apply is not None:
                fast = compiled_apply(
                    q,
                    k,
                    position_ids,
                    self.inv_freq,
                    self.position_selector,
                )
            if fast is not None:
                return fast

        cos, sin = self(k, position_ids)
        return apply_multimodal_rotary_pos_emb(
            q,
            k,
            cos,
            sin,
            mrope_section=self.mrope_section,
            unsqueeze_dim=unsqueeze_dim,
            style=self.style,
            cast_output=cast_output,
        )


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)


def rotate_half_even_odd(x):
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return mx.flatten(mx.stack([-x2, x1], axis=-1), start_axis=-2, end_axis=-1)


@mx.compile
def _apply_interleaved_rotary_pos_emb_axis1(q, k, cos, sin):
    cos = mx.expand_dims(cos, axis=1)
    sin = mx.expand_dims(sin, axis=1)

    rotary_dim = cos.shape[-1]
    q_rot = q[..., :rotary_dim]
    q_pass = q[..., rotary_dim:]
    k_rot = k[..., :rotary_dim]
    k_pass = k[..., rotary_dim:]

    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)
    q_embed = q_embed.astype(q.dtype)
    k_embed = k_embed.astype(k.dtype)

    return (
        mx.concatenate([q_embed, q_pass], axis=-1),
        mx.concatenate([k_embed, k_pass], axis=-1),
    )


def _section_cos_sin(cos, sin, mrope_section):
    split_indices = _cumulative_splits(list(mrope_section) * 2)
    cos = mx.concatenate(
        [m[i % 3] for i, m in enumerate(mx.split(cos, split_indices, axis=-1))],
        axis=-1,
    )[:, None, :, :]
    sin = mx.concatenate(
        [m[i % 3] for i, m in enumerate(mx.split(sin, split_indices, axis=-1))],
        axis=-1,
    )[:, None, :, :]
    return cos, sin


def _maybe_fast_precomputed_rotary(
    q,
    k,
    cos,
    sin,
    *,
    pairing: str,
    cos_layout: str = _HALF_COS,
    mrope_section: Optional[Sequence[int]] = None,
    unsqueeze_dim: int = 1,
    cast_output: bool = True,
):
    if (
        not _HAS_METAL
        or unsqueeze_dim != 1
        or not cast_output
        or q.ndim != 4
        or k.ndim != 4
        or cos.ndim not in (3, 4)
        or sin.ndim != cos.ndim
    ):
        return None

    rotary_dim = cos.shape[-1]
    if rotary_dim > q.shape[-1] or rotary_dim > k.shape[-1]:
        return None

    sectioned = mrope_section is not None
    position_selector = None
    if sectioned:
        if cos.ndim != 4:
            return None
        position_selector = mrope_position_selector(
            "chunked",
            mrope_section,
            rotary_dim // 2,
        )

    compiled_apply = _compiled_rotary_apply(
        rotary_dim,
        pairing,
        cos.ndim,
        sectioned,
        cos_layout,
    )
    if compiled_apply is None:
        return None

    return compiled_apply(q, k, cos, sin, position_selector)


def apply_rotary_pos_emb_even_odd(q, k, cos, sin, *, cos_layout: str = _HALF_COS):
    """Apply even/odd RoPE from precomputed cos/sin with a Metal fast path."""
    fast = _maybe_fast_precomputed_rotary(
        q,
        k,
        cos,
        sin,
        pairing=_EVEN_ODD,
        cos_layout=cos_layout,
    )
    if fast is not None:
        return fast

    cos = cos[:, None, :, :]
    sin = sin[:, None, :, :]
    if cos_layout == _HALF_COS:
        cos = mx.repeat(cos[..., : cos.shape[-1] // 2], repeats=2, axis=-1)
        sin = mx.repeat(sin[..., : sin.shape[-1] // 2], repeats=2, axis=-1)

    rotary_dim = cos.shape[-1]
    q_rot = q[..., :rotary_dim]
    q_pass = q[..., rotary_dim:]
    k_rot = k[..., :rotary_dim]
    k_pass = k[..., rotary_dim:]

    q_embed = (q_rot.astype(mx.float32) * cos) + (
        rotate_half_even_odd(q_rot).astype(mx.float32) * sin
    )
    k_embed = (k_rot.astype(mx.float32) * cos) + (
        rotate_half_even_odd(k_rot).astype(mx.float32) * sin
    )
    q_embed = q_embed.astype(q.dtype)
    k_embed = k_embed.astype(k.dtype)

    if q_pass.shape[-1] == 0 and k_pass.shape[-1] == 0:
        return q_embed, k_embed

    return (
        mx.concatenate([q_embed, q_pass], axis=-1),
        mx.concatenate([k_embed, k_pass], axis=-1),
    )


def apply_multimodal_rotary_pos_emb(
    q,
    k,
    cos,
    sin,
    *,
    mrope_section: Optional[Sequence[int]] = None,
    unsqueeze_dim: int = 1,
    style: str = "interleaved",
    cast_output: bool = True,
):
    if style == "interleaved" and unsqueeze_dim == 1 and cast_output:
        return _apply_interleaved_rotary_pos_emb_axis1(q, k, cos, sin)

    if style in {"sectioned_half_split", "sectioned_even_odd"}:
        if mrope_section is None:
            raise ValueError("mrope_section is required for sectioned MRoPE")
        fast = _maybe_fast_precomputed_rotary(
            q,
            k,
            cos,
            sin,
            pairing=_pairing_for_style(style),
            mrope_section=mrope_section,
            unsqueeze_dim=unsqueeze_dim,
            cast_output=cast_output,
        )
        if fast is not None:
            return fast
        cos, sin = _section_cos_sin(cos, sin, mrope_section)
    else:
        cos = mx.expand_dims(cos, axis=unsqueeze_dim)
        sin = mx.expand_dims(sin, axis=unsqueeze_dim)

    if style in {"sectioned_even_odd", "split_select"}:
        cos = mx.repeat(cos[..., : cos.shape[-1] // 2], repeats=2, axis=-1)
        sin = mx.repeat(sin[..., : sin.shape[-1] // 2], repeats=2, axis=-1)
        rotate_fn = rotate_half_even_odd
    else:
        rotate_fn = rotate_half

    rotary_dim = cos.shape[-1]
    q_rot = q[..., :rotary_dim]
    q_pass = q[..., rotary_dim:]
    k_rot = k[..., :rotary_dim]
    k_pass = k[..., rotary_dim:]

    q_embed = (q_rot * cos) + (rotate_fn(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_fn(k_rot) * sin)

    if cast_output:
        q_embed = q_embed.astype(q.dtype)
        k_embed = k_embed.astype(k.dtype)

    if q_pass.shape[-1] == 0 and k_pass.shape[-1] == 0:
        return q_embed, k_embed

    q_embed = mx.concatenate([q_embed, q_pass], axis=-1)
    k_embed = mx.concatenate([k_embed, k_pass], axis=-1)

    return q_embed, k_embed
