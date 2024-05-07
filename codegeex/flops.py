from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

from .parameters import calculate_num_parameters


def flops_per_param(
    num_layers: int,
    hidden_size: int,
    sequence_length: int,
    num_params: int,
) -> int:
    flops_per_token = 2 * num_params
    flops_per_seq = flops_per_token * sequence_length
    attn_flops_per_seq = num_layers * 2 * 2 * (hidden_size * (sequence_length**2))
    return flops_per_seq + attn_flops_per_seq


def estimate_training_flops(
    module: nn.Module,
    num_layers: int,
    hidden_size: int,
    sequence_length: int,
    gradient_accumulation_steps: int = 1,
):
    """
    Estimated FLOPs for MFU. Uses all parameters which leads to over-estimation
    because not all model parameters actually contribute to this FLOP
    computation. For this reason, the result will be higher by a fixed
    percentage (~10%) compared to the measured FLOPs.

    Refs:
        * https://ar5iv.labs.arxiv.org/html/2205.05198#A1
        * https://ar5iv.labs.arxiv.org/html/2204.02311#A2
    """
    num_trainable_params = calculate_num_parameters(module, requires_grad=True)
    num_frozen_params = calculate_num_parameters(module, requires_grad=False)

    # Each parameter is used in the forward and backward pass and during the
    # gradient computation.
    train_flops = flops_per_param(
        num_layers, hidden_size, sequence_length, num_trainable_params
    ) + (2 + (1 / gradient_accumulation_steps))

    # Frozen parameters are only used in the forward pass and during the
    # gradient computation.
    frozen_flops = (
        flops_per_param(num_layers, hidden_size, sequence_length, num_frozen_params) * 2
    )

    return train_flops + frozen_flops


def measure_flops(model: nn.Module, *args):
    flop_counter = FlopCounterMode(model, display=False)
    ctx = nullcontext() if model.training else torch.no_grad()
    with ctx, flop_counter:
        y = model(*args)
        if model.training:
            y.sum().backward()
    return flop_counter.get_total_flops()
