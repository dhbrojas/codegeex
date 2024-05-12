import math


def warmup_stable_decay(W, S, D, min_lr_scale_factor=0.1):
    """
    Returns a lambda function for PyTorch's LambdaLR scheduler implementing the
    WSD learning rate schedule.

    Parameters:
    - W: The last step of the warmup phase.
    - S: The last step of the stable phase.
    - D: The last step of the decay phase.
    - min_lr_scale_factor: The minimum learning rate is eta * min_lr_scale_factor.

    Returns:
    - A lambda function to be used with torch.optim.lr_scheduler.LambdaLR.
    """

    def lr_lambda(current_step):
        if current_step <= W:
            return current_step / W
        elif current_step <= S:
            return 1
        else:
            return (
                min_lr_scale_factor
                + (1 - min_lr_scale_factor)
                * (1 + math.cos(math.pi * (current_step - S) / (D - S)))
                / 2
            )

    return lr_lambda


def warmup_cosine_decay(W, D, min_lr_scale_factor=0.1):
    """
    Returns a lambda function for PyTorch's LambdaLR scheduler implementing the
    cosine decay learning rate schedule. It includes a warmup phase that lasts
    W steps and decays the learning rate to min_lr_scale_factor after D steps.
    After that the learning rate remains at eta * min_lr_scale_factor forever.

    Parameters:
    - W: The last step of the warmup phase.
    - D: The last step of the decay phase.
    - min_lr_scale_factor: The minimum learning rate is eta * min_lr_scale_factor.
    """

    def lr_lambda(current_step):
        if current_step <= W:
            return current_step / W
        elif current_step <= D:
            return (
                min_lr_scale_factor
                + (1 - min_lr_scale_factor)
                * (1 + math.cos(math.pi * (current_step - W) / (D - W)))
                / 2
            )
        else:
            return min_lr_scale_factor

    return lr_lambda
