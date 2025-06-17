import math
from collections.abc import Callable, Iterable

import torch


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logits_stable = logits - logits.max(dim=-1, keepdim=True).values
    logits_exp_sum = logits_stable.exp().sum(dim=-1, keepdim=True)
    loss = -logits_stable[torch.arange(logits.size(0)), targets] + torch.log(logits_exp_sum)

    return loss.mean()


def cosine_lr_schedule(t: int, lr_max: float, lr_min: float, t_w: int, t_c: int) -> float:
    if t < t_w:
        return t * lr_max / t_w

    if t <= t_c:
        return lr_min + 0.5 * (1 + math.cos((t - t_w) / (t_c - t_w) * math.pi)) * (lr_max - lr_min)

    return lr_min


def clip_grad_norm(parameters: Iterable[torch.nn.Parameter], max_norm: float, eps: float = 1e-6):
    parameters = list(parameters)
    gradients = [p.grad for p in parameters if p.grad is not None]
    total_norm = torch.linalg.vector_norm(torch.cat(gradients))

    for p in parameters:
        if p.grad is None:
            continue

        if total_norm > max_norm:
            p.grad *= max_norm / (total_norm + eps)


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
