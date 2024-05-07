import torch

from .flops import estimate_training_flops
from .parameters import calculate_num_parameters


def print_model_information(model, num_layers, hidden_size, sequence_length):
    num_parameters = calculate_num_parameters(model)
    print(f"Number of parameters: {num_parameters/1_000_000}M")
    estimated_flops = estimate_training_flops(
        model,
        num_layers=num_layers,
        hidden_size=hidden_size,
        sequence_length=sequence_length,
    )
    print(f"Estimated training FLOPS: {estimated_flops / 10**12:.2f} TFLOPS")
    print(f"Memory: {round(torch.cuda.memory_allocated()/(1024**3))}GiB")
