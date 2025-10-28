import math
from collections.abc import Callable, Iterable, Iterator

import torch
from jaxtyping import Float, Int


def cross_entropy(
    logits: Float[torch.Tensor, "batch_size vocab_size"], targets: Int[torch.Tensor, " batch_size"]
) -> Float[torch.Tensor, ""]:
    # Cross-entropy is just calculating entropy (the average surprise of a distribution)
    # using a wrong belief Q
    #
    # https://www.youtube.com/watch?v=tXE23653JrU is a good video to watch up on this.
    #
    # Given that:
    #   I(x) = \log\frac{1}{P(x)}
    #   H(P) = \sum_i P(x_i)\log\frac{1}{P(x_i)}
    #   H(P, Q) = \sum_i P(x_i)\log\frac{1}{Q(x_i)}
    #
    # Our wrong belief, Q, can be generated from the softmax of the logits.
    # The true distribution, P, is given to us by the targets.
    #
    # Since the true distribution P is a one-hot encoding of token that we want,
    # the cross-entropy calculation simplifies.
    #
    # H(P, Q) = \log\frac{1}{Q(x_i)} = -\log Q(x_i), where x_i is the target token
    #
    # We need to retrieve Q. Since we start off with the logits, we first need to convert them to a
    # probability distribution so that we can compare them with the targets, P. To do this,
    # we can just calculate Q using a numerically stable softmax on the logits.
    #
    # H(P, Q) = -\log Q(x_i)
    #         = -\log \frac{e^{o[x_i]}{\sum_a e^{o[a]}}
    #         = -\log e^{o[x_i]} + \log \sum_a e^{o[a]}
    #         = -o[x_i] + \log \sum_a e^{o[a]}
    logits_stable = logits - logits.max(dim=-1, keepdim=True).values
    logits_exp_sum = logits_stable.exp().sum(dim=-1, keepdim=True)
    loss = -logits_stable[torch.arange(logits.size(0)), targets] + torch.log(logits_exp_sum)

    # Calculate the loss across the entire batch.
    loss = loss.mean()

    return loss


def cosine_lr_schedule(t: int, lr_max: float, lr_min: float, t_w: int, t_c: int) -> float:
    # Here, we're just creating a function that returns the learning rate given that we're at
    # some time step t. The intuition is that during training, we want to use a bigger learning
    # rate in the beginning, and then slowly decay it to a smaller value.

    # On warm-up, we slowly reach a maximum learning rate
    if t < t_w:
        return t * lr_max / t_w

    # During the cosine annealing process, we slowly go down to the minimum learning rate
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

        # If the l_2 norm of the gradients exceeds a threshold, then scale it down
        if total_norm > max_norm:
            p.grad *= max_norm / (total_norm + eps)


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterator[torch.nn.Parameter],
        lr: float,
        weight_decay: float,
        betas: tuple[float, float],
        eps: float,
    ) -> None:
        defaults = {
            # The learning rate
            "lr": lr,
            # The weight decay rate, \lambda
            "weight_decay": weight_decay,
            # Hyperparamters, \beta_1, \beta_2, to control the updates to the moment estimates
            "betas": betas,
            # Epsilon, for numerical stability
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

                # Retrieve/initialize the state for AdamW, which includes keeping track of the
                # iteration, as well as the first and second moment estimates
                state = self.state[p]
                t = state.get("t", 1)
                m = state.get("m", torch.zeros_like(p.data))
                v = state.get("v", torch.zeros_like(p.data))

                # Update the first moment estimate
                m = b1 * m + (1 - b1) * p.grad.data

                # Update the second moment estimate
                v = b2 * v + (1 - b2) * torch.pow(p.grad.data, 2)

                # Calculate the iteration learning rate to update the parameters given the
                # moment estimates
                lr_t = lr * math.sqrt(1 - b2**t) / (1 - b1**t)
                p.data -= lr_t * m / (torch.sqrt(v) + eps)

                # Apply the weight decay
                p.data -= lr * weight_decay * p.data

                # Store the state for the next iteration
                state["t"] = t + 1
                state["m"] = m
                state["v"] = v

        return loss
