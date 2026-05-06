from typing import Optional, Sequence

import mlx.core as mx
import numpy as np


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
    split_indices = np.cumsum(mrope_section)[:-1].tolist()
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

    def __call__(self, x, position_ids):
        if position_ids.ndim == 2:
            position_ids = mx.broadcast_to(
                position_ids[None, ...],
                (3, position_ids.shape[0], position_ids.shape[1]),
            )

        inv_freq_expanded = mx.broadcast_to(
            self.inv_freq[None, None, :, None].astype(mx.float32),
            (3, position_ids.shape[1], self.inv_freq.shape[0], 1),
        )
        position_ids_expanded = position_ids[:, :, None, :].astype(mx.float32)

        freqs = inv_freq_expanded @ position_ids_expanded
        freqs = mx.swapaxes(freqs, 2, 3)
        freqs = apply_mrope_frequency_layout(
            freqs,
            self.mrope_section,
            style=self.style,
        )
        emb = mx.concatenate([freqs, freqs], axis=-1)
        cos = mx.cos(emb) * self.attention_scaling
        sin = mx.sin(emb) * self.attention_scaling

        if self.cast_output:
            return cos.astype(x.dtype), sin.astype(x.dtype)
        return cos, sin


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
    split_indices = np.cumsum(list(mrope_section) * 2)[:-1].tolist()
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
