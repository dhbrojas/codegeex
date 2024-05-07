import torch

from .flops import estimate_training_flops
from .parameters import calculate_num_parameters


def print_model_information(model, config):
    num_parameters = calculate_num_parameters(model)
    print(f"Number of parameters: {num_parameters/1_000_000}M")
    estimated_flops = estimate_training_flops(
        model,
        num_layers=config.num_layers,
        hidden_size=config.hidden_size,
        sequence_length=config.max_position_embeddings,
    )
    print(f"Estimated training FLOPS: {estimated_flops / 10**12:.2f} TFLOPS")
    print(f"Memory: {round(torch.cuda.memory_allocated()/(1024**3))}GiB")

    # model.eval()
    # measured_inference_flops = measure_flops(
    #     model, random_input(4, config.max_position_embeddings)
    # )
    # print(f"Measured inference FLOPS: {measured_inference_flops / 10**12:.2f} TFLOPS")

    # model.train()
    # measured_training_flops = measure_flops(
    #     model, random_input(4, config.max_position_embeddings)
    # )
    # print(f"Measured training FLOPS: {measured_training_flops / 10**12:.2f} TFLOPS")
