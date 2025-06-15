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
