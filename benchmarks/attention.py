"""
This script was imported and modified from https://github/Dao-AILab/flash-attention.
"""

import math

import torch
import torch.nn.functional as F
from einops import rearrange
from flash_attn import flash_attn_qkvpacked_func
from flash_attn.utils.benchmark import benchmark_fwd_bwd
from triton.ops.flash_attention import attention as attention_triton

try:
    import xformers.ops as xops
except ImportError:
    print("Warning: xformers not found")
    xops = None


def flops(batch, seqlen, headdim, nheads, causal, mode="fwd"):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)


def efficiency(flop, time):
    return (flop / time / 10**12) if not math.isnan(time) else 0.0


def attention_pytorch(qkv, dropout_p=0.0, causal=True):
    """
    Arguments:
        qkv: (batch_size, seqlen, 3, nheads, head_dim)
        dropout_p: float
    Output:
        output: (batch_size, seqlen, nheads, head_dim)
    """
    batch_size, seqlen, _, nheads, d = qkv.shape
    q, k, v = qkv.unbind(dim=2)
    q = rearrange(q, "b t h d -> (b h) t d")
    k = rearrange(k, "b s h d -> (b h) d s")
    softmax_scale = 1.0 / math.sqrt(d)
    # Preallocate attn_weights for `baddbmm`
    scores = torch.empty(
        batch_size * nheads, seqlen, seqlen, dtype=qkv.dtype, device=qkv.device
    )
    scores = rearrange(
        torch.baddbmm(scores, q, k, beta=0, alpha=softmax_scale),
        "(b h) t s -> b h t s",
        h=nheads,
    )
    if causal:
        # "triu_tril_cuda_template" not implemented for 'BFloat16'
        # So we have to construct the mask in float
        causal_mask = torch.triu(
            torch.full((seqlen, seqlen), -10000.0, device=scores.device), 1
        )
        # TD [2022-09-30]: Adding is faster than masked_fill_ (idk why, just better kernel I guess)
        scores = scores + causal_mask.to(dtype=scores.dtype)
    attention = torch.softmax(scores, dim=-1)
    attention_drop = F.dropout(attention, dropout_p)
    output = torch.einsum("bhts,bshd->bthd", attention_drop, v)
    return output.to(dtype=qkv.dtype)


def time_fwd_bwd(func, *args, **kwargs):
    time_f, time_b = benchmark_fwd_bwd(func, *args, **kwargs)
    return time_f[1].mean, time_b[1].mean


N = 30
DEVICE = "cuda"
DTYPE = torch.float16

BATCH_SIZES = [(32, 512), (16, 1024), (8, 2048), (4, 4096), (2, 8192), (1, 16384)]
CAUSAL = [False, True]
HEAD_DIMENSIONS = [64, 128]
HIDDEN_SIZES = [1024, 2048]
DROPOUT = 0.0

methods = (
    ["flash-attn-2", "torch", "triton"]
    + (["xformers-cutlass"] if xops is not None else [])
    + (["xformers-flash"] if xops is not None else [])
)

# Measurements
forward = {}
backward = {}
forward_backward = {}

# FLOPS
speed_forward = {}
speed_backward = {}
speed_forward_backward = {}


def grid(*args):
    if len(args) == 1:
        for x in args[0]:
            yield (x,)
    else:
        for x in args[0]:
            for y in grid(*args[1:]):
                yield (x,) + y


def make_qkv(batch_size, seqlen, headdim, nheads):
    return torch.randn(
        batch_size,
        seqlen,
        3,
        nheads,
        headdim,
        device=DEVICE,
        dtype=DTYPE,
        requires_grad=True,
    )


def make_q_k_v(batch_size, seqlen, headdim, nheads):
    return [
        torch.randn(
            batch_size,
            seqlen,
            nheads,
            headdim,
            device=DEVICE,
            dtype=DTYPE,
            requires_grad=True,
        )
        for _ in range(3)
    ]


def handle_exception(e, config, method):
    forward[config, method] = float("nan")
    backward[config, method] = float("nan")
    print(f"{method}: {config}: {e}")


for config in grid(CAUSAL, HIDDEN_SIZES, HEAD_DIMENSIONS, BATCH_SIZES):
    causal, hidden_size, head_dimension, (batch_size, sequence_length) = config  # type: ignore

    assert hidden_size % head_dimension == 0
    num_heads = hidden_size // head_dimension

    # Flash Attention 2
    qkv = make_qkv(batch_size, sequence_length, head_dimension, num_heads)

    try:
        f, b = time_fwd_bwd(
            flash_attn_qkvpacked_func,
            qkv,
            DROPOUT,
            causal=causal,
            repeats=N,
            verbose=False,
        )
        forward[config, "flash-attn-2"] = f
        backward[config, "flash-attn-2"] = b
    except Exception as e:
        handle_exception(e, config, "flash-attn-2")

    # PyTorch
    qkv = qkv.detach().requires_grad_(True)

    try:
        f, b = time_fwd_bwd(
            attention_pytorch,
            qkv,
            DROPOUT,
            causal=causal,
            repeats=N,
            verbose=False,
        )
        forward[config, "torch"] = f
        backward[config, "torch"] = b
    except Exception as e:
        handle_exception(e, config, "torch")

    # Triton
    q, k, v = [
        torch.randn(
            batch_size,
            num_heads,
            sequence_length,
            head_dimension,
            device=DEVICE,
            dtype=DTYPE,
            requires_grad=True,
        )
        for _ in range(3)
    ]

    try:
        f, b = time_fwd_bwd(
            attention_triton,
            q,
            k,
            v,
            causal,
            head_dimension ** (-0.5),
            False,
            repeats=N,
            verbose=False,
        )
        forward[config, "triton"] = f
        backward[config, "triton"] = b
    except Exception as e:
        handle_exception(e, config, "triton")

    if xops is not None:
        q, k, v = make_q_k_v(batch_size, sequence_length, head_dimension, num_heads)

        # XFormers Cutlass
        try:
            f, b = time_fwd_bwd(
                xops.memory_efficient_attention,
                q,
                k,
                v,
                attn_bias=xops.LowerTriangularMask() if causal else None,
                op=(xops.fmha.cutlass.FwOp, xops.fmha.cutlass.BwOp),
                verbose=False,
            )
            forward[config, "xformers-cutlass"] = f
            backward[config, "xformers-cutlass"] = b
        except Exception as e:
            handle_exception(e, config, "xformers-cutlass")

        [q, k, v] = map(lambda x: x.detach().requires_grad_(True), [q, k, v])

        # XFormers Flash
        try:
            f, b = time_fwd_bwd(
                xops.memory_efficient_attention,
                q,
                k,
                v,
                attn_bias=xops.LowerTriangularMask() if causal else None,
                op=(xops.fmha.flash.FwOp, xops.fmha.flash.BwOp),
                verbose=False,
            )
            forward[config, "xformers-flash"] = f
            backward[config, "xformers-flash"] = b
        except Exception as e:
            handle_exception(e, config, "xformers-flash")

    print(
        f">{' Causal,' if causal else ''} Hidden Size ({hidden_size}), Head Dimension ({head_dimension}), Num Heads ({num_heads}), Batch ({batch_size}, {sequence_length})"
    )
    for method in methods:
        forward_backward[config, method] = (
            forward[config, method] + backward[config, method]
        )
        speed_forward[config, method] = efficiency(
            flops(
                batch_size,
                sequence_length,
                head_dimension,
                num_heads,
                causal,
                mode="fwd",
            ),
            forward[config, method],
        )
        speed_backward[config, method] = efficiency(
            flops(
                batch_size,
                sequence_length,
                head_dimension,
                num_heads,
                causal,
                mode="bwd",
            ),
            backward[config, method],
        )
        speed_forward_backward[config, method] = efficiency(
            flops(
                batch_size,
                sequence_length,
                head_dimension,
                num_heads,
                causal,
                mode="fwd_bwd",
            ),
            forward_backward[config, method],
        )
        print(f"  - {method}")
        print(
            f"    - Forward:            {speed_forward[config, method]:>8.2f} TFLOPs/s"
        )
        print(
            f"    - Backward:           {speed_backward[config, method]:>8.2f} TFLOPs/s"
        )
        print(
            f"    - Forward + Backward: {speed_forward_backward[config, method]:>8.2f} TFLOPs/s"
        )
