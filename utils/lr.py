import math


def wsd_learning_rate_scheduler(W, S, D, min_lr_scale_factor=0.1):
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
