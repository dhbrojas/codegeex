import time

import torch
from accelerate import Accelerator
from torch.optim import AdamW

from models.codegeexnano import CodeGeeXNanoConfig, CodeGeeXNanoForCausalLM

# Benchmark the model
MICRO_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 256
STEPS = 2


def random_input(batch_size, vocab_size, sequence_length, device: torch.device):
    return torch.randint(
        0,
        vocab_size,
        (batch_size, sequence_length),
        device=device,
    )


def main():
    accelerator = Accelerator()

    config = CodeGeeXNanoConfig(
        hidden_size=1024,
        num_layers=20,
        intermediate_size=4096,
        max_position_embeddings=2048,
        norm_eps=1e-5,
        num_attention_heads=16,
        vocab_size=32768,
    )

    model = CodeGeeXNanoForCausalLM(config)
    model.init_weights()
    accelerator.print("Model initialized")

    optimizer = AdamW(model.parameters(), lr=1e-4)
    accelerator.print("Optimizer initialized")

    def random_input_generator(n, batch_size):
        for _ in range(n):
            yield (
                random_input(
                    batch_size,
                    config.vocab_size,
                    config.max_position_embeddings,
                    accelerator.device,
                ),
                random_input(
                    batch_size,
                    config.vocab_size,
                    config.max_position_embeddings,
                    accelerator.device,
                ),
            )

    model, optimizer = accelerator.prepare(model, optimizer)

    accelerator.print(f"Dtype: {accelerator.mixed_precision}")
    accelerator.print(f"Device: {accelerator.device}")

    step_loss = 0.0
    step_start = time.perf_counter()

    for i, (inputs, targets) in enumerate(
        random_input_generator(STEPS * GRADIENT_ACCUMULATION_STEPS, MICRO_BATCH_SIZE)
    ):
        loss = model(inputs, targets)
        accelerator.backward(loss)

        step_loss += loss.detach().item()

        if (i + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()

            step_duration = time.perf_counter() - step_start
            step_start = time.perf_counter()
            processed_tokens = (
                MICRO_BATCH_SIZE
                * config.max_position_embeddings
                * GRADIENT_ACCUMULATION_STEPS
            )

            step_loss /= GRADIENT_ACCUMULATION_STEPS

            accelerator.print(
                f"Step {i + 1}: Loss: {step_loss:.2f}, Duration: {step_duration:.2f}s, Tokens/s: {processed_tokens / step_duration:.2f}"
            )


if __name__ == "__main__":
    main()
