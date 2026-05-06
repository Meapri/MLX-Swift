from functools import lru_cache
from typing import Optional, Sequence

import mlx.core as mx

_HAS_METAL = mx.metal.is_available()


def _cumulative_splits(lengths: Sequence[int]):
    return mx.cumsum(mx.array(lengths, dtype=mx.int32))[:-1]


def _interleaved_position_selector(mrope_section: Sequence[int], freq_dim: int):
    selector = [0] * freq_dim
    for dim, offset in enumerate((1, 2), start=1):
        for idx in range(offset, min(mrope_section[dim] * 3, freq_dim), 3):
            selector[idx] = dim
    return mx.array(selector, dtype=mx.int32)


@mx.compile
def _interleaved_mrope_freqs(position_ids, inv_freq, position_selector):
    positions = mx.take(position_ids, position_selector, axis=0).transpose(1, 2, 0)
    return positions.astype(mx.float32) * inv_freq


@lru_cache(maxsize=None)
def _interleaved_mrope_apply_kernel(rotary_dim: int, position_ndim: int):
    if not _HAS_METAL:
        return None

    if position_ndim == 2:
        position_expr = "position_ids[b * q_len + t]"
    else:
        position_expr = "position_ids[(axis * q_bsz + b) * q_len + t]"

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
        name=f"interleaved_mrope_apply_{rotary_dim}_{position_ndim}d",
        input_names=["q", "k", "position_ids", "inv_freq", "position_selector"],
        output_names=["q_out", "k_out"],
        source=source,
    )


def _fast_interleaved_mrope_apply(
    q,
    k,
    position_ids,
    inv_freq,
    position_selector,
    rotary_dim: int,
):
    kernel = _interleaved_mrope_apply_kernel(rotary_dim, position_ids.ndim)
    if kernel is None:
        return None

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


def _apply_chunked_mrope(freqs, mrope_section):
    freqs_t = freqs[0]
    offset = mrope_section[0]
    for dim, length in enumerate(mrope_section[1:], start=1):
        idx = slice(offset, offset + length)
        freqs_t[..., idx] = freqs[dim, ..., idx]
        offset += length
    return freqs_t


def _apply_interleaved_mrope(freqs, mrope_section):
    freqs_t = freqs[0]
    for dim, offset in enumerate((1, 2), start=1):
        length = mrope_section[dim] * 3
        idx = slice(offset, length, 3)
        freqs_t[..., idx] = freqs[dim, ..., idx]
    return freqs_t


def _apply_split_select_mrope(freqs, mrope_section):
    split_indices = _cumulative_splits(mrope_section)
    chunks = mx.split(freqs, split_indices, axis=-1)
    return mx.concatenate([chunk[i % 3] for i, chunk in enumerate(chunks)], axis=-1)


def apply_mrope_frequency_layout(
    freqs,
    mrope_section: Sequence[int],
    *,
    style: str = "interleaved",
):
    mrope_section = list(mrope_section)

    if style == "chunked":
        return _apply_chunked_mrope(freqs, mrope_section)
    if style == "interleaved":
        return _apply_interleaved_mrope(freqs, mrope_section)
    if style == "split_select":
        return _apply_split_select_mrope(freqs, mrope_section)
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

    if style == "interleaved":
        if position_selector is None:
            position_selector = _interleaved_position_selector(
                mrope_section, inv_freq.shape[0]
            )
        return _interleaved_mrope_freqs(position_ids, inv_freq, position_selector)

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
        self.position_selector = (
            _interleaved_position_selector(self.mrope_section, self.inv_freq.shape[0])
            if style == "interleaved"
            else None
        )
        self.fused_apply = style == "interleaved" and _HAS_METAL

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
            fast = _fast_interleaved_mrope_apply(
                q,
                k,
                position_ids,
                self.inv_freq,
                self.position_selector,
                self.dim,
            )
            if fast is not None:
                return fast

        cos, sin = self(k, position_ids)
        return apply_multimodal_rotary_pos_emb(
            q,
            k,
            cos,
            sin,
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
        cos, sin = _section_cos_sin(cos, sin, mrope_section)
    else:
        cos = mx.expand_dims(cos, axis=unsqueeze_dim)
        sin = mx.expand_dims(sin, axis=unsqueeze_dim)

    if style == "sectioned_even_odd":
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
