import os

import torch
import torch.distributed as dist
from tensorboardX import SummaryWriter


class MetricsReporter:
    def __init__(self, logdir: str):
        os.makedirs(logdir, exist_ok=True)
        self.writer = (
            SummaryWriter(logdir)
            if (dist.is_initialized() and dist.get_rank() == 0)
            else None
        )

    def record(self, tag: str, v: float, i: int):
        if self.writer:
            self.writer.add_scalar(tag, v, i)

    def flush(self):
        if self.writer:
            self.writer.flush()


class DistMetric:
    def __init__(self, device: torch.device):
        self.values = []
        self.device = device

    def add(self, v):
        self.values.append(v)

    def sum(self):
        """Sum uses distributed sum reduction and flushes values."""
        t = torch.tensor(self.values, device=self.device)
        dist.all_reduce(t, dist.ReduceOp.SUM)
        self.values = []
        return t.item()

    def mean(self):
        """Mean uses distributed mean reduction and flushes values."""
        t = torch.tensor(self.values, device=self.device)
        dist.all_reduce(t, dist.ReduceOp.SUM)
        t /= dist.get_world_size()
        self.values = []
        return t.item()

    def all(self):
        """All returns a Tensor with all the values from all processes."""
        out = [
            torch.zeros(len(self.values), device=self.device)
            for _ in range(dist.get_world_size())
        ]
        dist.all_gather(out, torch.tensor(self.values, device=self.device))
        self.values = []
        out = torch.cat(out).flatten()
        return out.tolist()
