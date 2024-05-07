def wsd_learning_rate_scheduler(W, T, eta, decay_rate=0.95, decay_step=1):
    """
    Returns a lambda function for PyTorch's LambdaLR scheduler implementing the WSD learning rate schedule.

    Parameters:
    - W: The last step of the warmup phase.
    - T: The last step of the stable phase.
    - eta: The maximum learning rate (used during the stable phase).
    - decay_rate: The decay rate of the learning rate for each decay_step after T.
    - decay_step: The step interval for applying the decay_rate.

    Returns:
    - A lambda function to be used with torch.optim.lr_scheduler.LambdaLR.
    """

    def lr_lambda(current_step):
        if current_step <= W:
            # Linear warmup
            return current_step / W * eta
        elif current_step <= T:
            # Stable learning rate
            return eta
        else:
            # Exponential decay
            steps_into_decay = current_step - T
            decay_factor = decay_rate ** (steps_into_decay / decay_step)
            return eta * decay_factor

    return lr_lambda
