import math
import sys
from dataclasses import dataclass
from functools import partial
from typing import Optional, Tuple

import torch
import torch.nn as nn
from flash_attn import flash_attn_qkvpacked_func
from xformers.ops import SwiGLU

sys.path.append("..")

from utils.rope import ApplyRoPE

RoPECache = Tuple[torch.Tensor, torch.Tensor]


# TODO: Add padded vocab size.
@dataclass
class CodeGeeXNanoConfig:
    # Size of the hidden layer in the transformer encoder
    hidden_size: int = 1024
    # Size of the intermediate layer in the feedforward network
    intermediate_size: int = 4096
    # Number of layers in the transformer encoder
    num_layers: int = 24
    # Size of the vocabulary
    vocab_size: int = 32768
    # Number of attention heads in the multi-head attention mechanism
    num_attention_heads: int = 16
    # Maximum length of the input sequence
    max_position_embeddings: int = 4096
    # Epsilon value for layer normalization
    norm_eps: float = 1e-5
    # RoPE base value
    rope_theta: int = 10000
    # RoPE scaling ratio
    rope_scaling_ratio: int = 1
    # The percentage of head dimensions used for RoPE
    rope_percentage: float = 0.25

    def __post_init__(self):
        assert self.hidden_size % self.num_attention_heads == 0, (
            f"hidden_size ({self.hidden_size}) must be divisible by "
            f"num_attention_heads ({self.num_attention_heads})"
        )
        assert (
            self.rope_percentage > 0 and self.rope_percentage <= 1
        ), f"rope_percentage ({self.rope_percentage}) must be in the range (0, 1]"

    @property
    def head_size(self):
        return self.hidden_size // self.num_attention_heads


class CodeGeeXNanoForCausalLM(nn.Module):
    def __init__(self, config: CodeGeeXNanoConfig):
        super().__init__()
        self.config = config
        self.model = CodeGeeXNanoModel(config)

    def init_weights(self):
        self.model.apply(partial(self.model.init_weights, depth=self.config.num_layers))

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        outputs = self.model(inputs)
        batch_size, sequence_length, vocab_size = outputs.size()

        outputs = outputs.view(-1, vocab_size).float()
        targets = targets.view(-1)

        return nn.functional.cross_entropy(
            outputs,
            targets,
        )


class CodeGeeXNanoModel(nn.Module):
    """
    CodeGeeXNano is a decoder-only transformer model with RoPE, multi-head
    attention, tied input and output embeddings, RMS pre-norm and no KQV,
    attention and MLP bias.
    """

    def __init__(self, config: CodeGeeXNanoConfig):
        super().__init__()
        self.config = config
        self.rope: Optional[RoPECache] = None
        self.transformer = nn.ModuleDict(
            dict(
                embeddings=nn.Embedding(config.vocab_size, config.hidden_size),
                layers=nn.ModuleList(
                    CodeGeeXNanoBlock(config) for _ in range(config.num_layers)
                ),
            )
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, inputs: torch.Tensor):
        batch_size, sequence_length = inputs.size()
        assert sequence_length <= self.config.max_position_embeddings

        x = self.transformer.embeddings(inputs)
        batch_size, sequence_length, hidden_size = x.size()
        assert hidden_size == self.config.hidden_size

        if (
            self.rope is None
            or self.rope[0].device != x.device
            or self.rope[0].dtype != x.dtype
        ):
            self.rope = self.build_rope_cache(
                x.device,
                x.dtype,
            )

        cos, sin = self.rope
        cos, sin = cos[:sequence_length], sin[:sequence_length]

        for layer in self.transformer.layers:
            x = layer(x, self.rope)

        x = self.lm_head(x)
        batch_size, sequence_length, vocab_size = x.size()
        assert vocab_size == self.config.vocab_size

        return x

    def init_weights(self, module: nn.Module, depth: int):
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(
                module.weight,
                mean=0.0,
                std=math.sqrt(2.0 / 5 / self.config.hidden_size),
            )
        elif isinstance(module, nn.Linear):
            torch.nn.init.normal_(
                module.weight,
                mean=0.0,
                std=math.sqrt(2.0 / 5 / self.config.hidden_size),
            )
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        for name, p in module.named_parameters():
            if (
                name == "proj.weight" and isinstance(module, CodeGeeXNanoFeedForward)
            ) or (
                name == "w3.weight"
                and isinstance(module, SwiGLU)
                or (
                    name == "proj.weight"
                    and isinstance(module, CodeGeeXNanoMultiHeadCausalFlashAttention)
                )
            ):
                nn.init.normal_(
                    p, mean=0.0, std=1 / math.sqrt(self.config.hidden_size) / depth
                )

    def build_rope_cache(self, device: torch.device, dtype: torch.dtype):
        theta = 1.0 / (
            self.config.rope_theta
            ** (
                torch.arange(
                    0,
                    self.config.rope_percentage * self.config.head_size,
                    2,
                    device=device,
                )
                / self.config.hidden_size
            )
        )

        indices = (
            torch.arange(self.config.max_position_embeddings, device=device)
            / self.config.rope_scaling_ratio
        )

        idx_theta = torch.outer(indices, theta)

        cos, sin = torch.cos(idx_theta), torch.sin(idx_theta)

        # Note from the original implementation:
        # > Added to ensure same data type with queries and keys, to use fused
        # > rotary embedding
        if dtype == torch.bfloat16:
            return cos.bfloat16(), sin.bfloat16()
        # Note from the original implementation:
        # > This is to mimic the behaviour of complex32, else we will get
        # > different results
        if dtype in (torch.float16, torch.bfloat16, torch.int8):
            return cos.half(), sin.half()

        return cos, sin


class CodeGeeXNanoMultiHeadCausalFlashAttention(nn.Module):
    def __init__(self, config: CodeGeeXNanoConfig):
        super().__init__()
        self.config = config

        self.attention = nn.Linear(
            self.config.hidden_size,
            self.config.hidden_size * 3,
            bias=False,
        )
        self.projection = nn.Linear(
            self.config.hidden_size, self.config.hidden_size, bias=False
        )

    def forward(self, x: torch.Tensor, rope: torch.Tensor):
        batch_size, sequence_length, hidden_size = x.size()
        assert hidden_size == self.config.hidden_size
        assert rope[0].device == x.device

        x = self.attention(x)
        x = x.reshape(
            batch_size,
            sequence_length,
            3,
            self.config.num_attention_heads,
            self.config.head_size,
        )

        cos, sin = rope

        q, k, v = x.unbind(dim=2)
        q = ApplyRoPE.apply(q.type_as(cos), cos, sin).type_as(v)  # type: ignore
        k = ApplyRoPE.apply(k.type_as(cos), cos, sin).type_as(v)  # type: ignore

        x = torch.stack((q, k, v), dim=2)

        x = self.scaled_dot_product_attention(x)
        batch_size, sequence_length, num_heads, head_size = x.size()
        assert num_heads == self.config.num_attention_heads

        x = x.reshape(batch_size, sequence_length, -1)
        x = self.projection(x)

        return x

    def scaled_dot_product_attention(self, x: torch.Tensor):
        batch_size, sequence_length, _, num_heads, head_size = x.size()
        assert (
            num_heads == self.config.num_attention_heads
        ), f"{num_heads} != {self.config.num_attention_heads} (x.shape={x.shape})"
        assert (
            head_size == self.config.head_size
        ), f"{head_size} != {self.config.head_size} (x.shape={x.shape})"

        return flash_attn_qkvpacked_func(x, causal=True)


class CodeGeeXNanoFeedForward(nn.Module):
    def __init__(self, config: CodeGeeXNanoConfig):
        super().__init__()
        self.swiglu = SwiGLU(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            # Performance decreases slightly when using packed weights.
            _pack_weights=False,
        )

    def forward(self, x: torch.Tensor):
        return self.swiglu(x)


class CodeGeeXNanoRMSNorm(nn.Module):
    """Same normalization as in Google Gemma."""

    def __init__(self, config: CodeGeeXNanoConfig):
        super().__init__()
        self.config = config
        self.weight = nn.Parameter(torch.zeros(config.hidden_size))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(
            x.square().mean(dim=-1, keepdim=True) + self.config.norm_eps
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = self.normalize(x.float())
        norm = norm * (1.0 + self.weight.float())
        return norm.type_as(x)


class CodeGeeXNanoBlock(nn.Module):
    def __init__(self, config: CodeGeeXNanoConfig):
        super().__init__()
        self.config = config
        self.pre_norm = CodeGeeXNanoRMSNorm(config)
        self.attention = CodeGeeXNanoMultiHeadCausalFlashAttention(config)
        self.post_attention_norm = CodeGeeXNanoRMSNorm(config)
        self.feedforward = CodeGeeXNanoFeedForward(config)

    def forward(self, x: torch.Tensor, rope: RoPECache):
        batch_size, sequence_length, hidden_size = x.size()
        assert hidden_size == self.config.hidden_size

        residual = x

        x = self.pre_norm(x)
        x = self.attention(x, rope) + residual
        x = self.post_attention_norm(x)
        x = self.feedforward(x) + residual

        return x
