import os
import typing

import numpy as np
import numpy.typing as npt
import torch


def load_data(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = []
    targets = []

    max_start_idx = len(dataset) - context_length - 1
    start_indices = np.random.randint(0, max_start_idx + 1, size=batch_size)
    for start_idx in start_indices:
        inputs.append(dataset[start_idx : start_idx + context_length])
        targets.append(dataset[start_idx + 1 : start_idx + context_length + 1])

    input_tensor = torch.tensor(np.array(inputs), dtype=torch.long, device=device)
    target_tensor = torch.tensor(np.array(targets), dtype=torch.long, device=device)

    return input_tensor, target_tensor


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
) -> None:
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }

    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    checkpoint: dict = torch.load(src)

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint["iteration"]
