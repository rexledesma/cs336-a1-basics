import math
from collections.abc import Callable

import torch


def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    inputs_max = torch.max(inputs, dim=-1, keepdim=True)[0]
    inputs = inputs - inputs_max

    exp_inputs = torch.exp(inputs)
    sum_exp = torch.sum(exp_inputs, dim=-1, keepdim=True)
    log_probs = inputs - torch.log(sum_exp)

    batch_size = inputs.shape[0]
    target_log_probs = log_probs[torch.arange(batch_size), targets]

    loss = -torch.mean(target_log_probs)

    return loss


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr: float, weight_decay: float, betas: tuple[float, float], eps: float) -> None:
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "betas": betas,
            "eps": eps,
        }
        super().__init__(params, defaults)

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]

                t = state.get("t", 1)
                m = state.get("m", torch.zeros_like(p.data))
                v = state.get("v", torch.zeros_like(p.data))
                grad = p.grad.data

                m = b1 * m + (1 - b1) * grad
                v = b2 * v + (1 - b2) * torch.pow(grad, 2)
                lr_t = lr * math.sqrt(1 - b2**t) / (1 - b1**t)

                p.data -= lr_t * m / (torch.sqrt(v) + eps)
                p.data -= lr * weight_decay * p.data

                state["t"] = t + 1
                state["m"] = m
                state["v"] = v

        return loss
