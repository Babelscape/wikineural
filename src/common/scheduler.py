from torch.optim.lr_scheduler import LambdaLR

from math import sqrt


# this is how fairseq implements it
# https://fairseq.readthedocs.io/en/latest/lr_scheduler.html
def inverse_sqrt_lr_lambda(
    warmup_steps: int,
    base_lr: float = None,
    warmup_init_lr: float = None,
    min_lr: float = None,
):

    # we are using a LambdaLR, which multiplies the base_lr by some coefficient returned by _lambda
    # thus, we have to normalize all learning rates so as to become scalar multipliers

    # handle case in which no base_lr is provided
    if base_lr is None:
        assert warmup_init_lr is None and min_lr is None
        base_lr = 1.0

    warmup_init_lr = warmup_init_lr or base_lr
    init_multiplier = warmup_init_lr / base_lr
    warmup_step_size = (1 - init_multiplier) / warmup_steps
    decay = sqrt(warmup_steps)

    min_multiplier = min_lr / base_lr if min_lr else None

    def _lambda(step: int) -> float:
        if step < warmup_steps:
            return init_multiplier + step * warmup_step_size

        value = decay / sqrt(step)

        if min_multiplier and min_multiplier > value:
            value = min_multiplier

        return value

    return _lambda


def inverse_sqrt_lr_scheduler(
    optimizer,
    warmup_steps: int,
    base_lr: float = None,
    warmup_init_lr: float = None,
    min_lr: float = None,
):
    return LambdaLR(
        optimizer, inverse_sqrt_lr_lambda(warmup_steps, base_lr, warmup_init_lr, min_lr)
    )


class InverseSQRTScheduler(LambdaLR):
    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        base_lr: float = None,
        warmup_init_lr: float = None,
        min_lr: float = None,
    ):
        super().__init__(
            optimizer,
            inverse_sqrt_lr_lambda(warmup_steps, base_lr, warmup_init_lr, min_lr),
        )


def plot_lr_scheduler(base_lr, _lambda, steps=(0, 100000)):
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(*steps)
    y = base_lr * np.vectorize(_lambda)(x)
    plt.plot(x, y)
    plt.savefig("lr.png")


if __name__ == "__main__":
    _min_lr = 1e-9
    _warmup_steps = 150
    _base_lr = 0.0005
    _warmup_init_lr = 1e-7
    inv_sqrt_lambda = inverse_sqrt_lr_lambda(
        warmup_steps=_warmup_steps,
        base_lr=_base_lr,
        warmup_init_lr=_warmup_init_lr,
        min_lr=_min_lr,
    )
    plot_lr_scheduler(_base_lr, inv_sqrt_lambda, steps=(0, 200))
