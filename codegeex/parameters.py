import torch.nn as nn


def calculate_num_parameters(model: nn.Module, requires_grad: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
