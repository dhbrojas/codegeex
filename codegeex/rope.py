# Adapted from https://github.com/dao-ailab/flash-attention
# Copyright (c) 2023, Tri Dao.

import rotary_emb
import torch
from einops import rearrange
from torch.autograd import Function


class ApplyRoPE(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Function,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ):
        """
        Applies the rotary position encoding to the input tensor in-place.

        Args:
            x: (batch_size, sequence_length, num_heads, head_size)
            cos, sin: (sequence_length, rotary_dim / 2)
        """
        batch_size, sequence_length, num_heads, head_size = x.shape
        rotary_sequence_length, rotary_size = cos.shape
        rotary_size *= 2

        assert rotary_size <= head_size
        assert sequence_length <= rotary_sequence_length
        assert sin.shape == (rotary_sequence_length, rotary_size // 2)

        x_rope = x[..., :rotary_size]
        x1_rope, x2_rope = x_rope.chunk(2, dim=-1)

        assert x1_rope.device == cos.device, f"{x1_rope.device} != {cos.device}"
        assert x1_rope.dtype == cos.dtype, f"{x1_rope.dtype} != {cos.dtype}"

        rotary_emb.apply_rotary(
            x1_rope,
            x2_rope,
            rearrange(cos[:sequence_length], "s d -> s 1 d"),
            rearrange(sin[:sequence_length], "s d -> s 1 d"),
            x1_rope,
            x2_rope,
            False,
        )

        ctx.save_for_backward(cos, sin)

        return x

    @staticmethod
    def backward(ctx: Function, dy: torch.Tensor):
        cos, sin = ctx.saved_tensors  # type: ignore

        batch_size, sequence_length, num_heads, head_size = dy.shape
        rotary_size = cos.shape[-1]
        rotary_size *= 2

        dy_rope = dy[..., :rotary_size]
        dy_rope_1, dy_rope_2 = dy_rope.chunk(2, dim=-1)

        rotary_emb.apply_rotary(
            dy_rope_1,
            dy_rope_2,
            rearrange(cos[:sequence_length], "s d -> s 1 d"),
            rearrange(sin[:sequence_length], "s d -> s 1 d"),
            dy_rope_1,
            dy_rope_2,
            True,
        )

        return dy, None, None, None, None
