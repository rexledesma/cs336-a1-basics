import pickle
import time
from itertools import islice
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import typer

import wandb
from cs336_basics.optimizer import AdamW, clip_grad_norm, cosine_lr_schedule, cross_entropy
from cs336_basics.tokenizer import BPETokenizer
from cs336_basics.train import load_data, save_checkpoint
from cs336_basics.train_bpe import train_bpe
from cs336_basics.transformer import Transformer, softmax

app = typer.Typer()

ROOT_DIRECTORY_PATH = (
    Path(__file__)
    .joinpath(
        "..",
        "..",
    )
    .resolve()
)


@app.command()
def create_bpe(input_path: Path, vocab_size: int):
    """Train BPE on a text file and serialize the vocabulary and merges."""
    start = time.perf_counter()
    vocabulary, merges = train_bpe(input_path, vocab_size, ["<|endoftext|>"])
    duration = time.perf_counter() - start

    typer.echo(f"Vocabulary size: {len(vocabulary)}")
    typer.echo(f"Training took {duration:.2f} seconds")

    vocab_path = input_path.with_suffix(".vocab.pkl")
    merges_path = input_path.with_suffix(".merges.pkl")

    # Serialize vocabulary and merges
    vocab_path.unlink(missing_ok=True)
    merges_path.unlink(missing_ok=True)

    pickle.dump(vocabulary, vocab_path.open("wb"))
    pickle.dump(merges, merges_path.open("wb"))

    typer.echo(f"Serialized vocabulary to {vocab_path}")
    typer.echo(f"Serialized merges to {merges_path}")


@app.command()
def run_bpe(vocab_path: Path, merges_path: Path, input_path: Path, num_lines: int | None = None):
    """Create a BPETokenizer from pickled vocabulary and merges files."""
    tokenizer = BPETokenizer.from_files(vocab_path, merges_path, ["<|endoftext|>"])

    typer.echo("Successfully created BPETokenizer")
    typer.echo(f"Vocabulary size: {len(tokenizer.token_for_id)}")

    input_file = input_path.open("r")
    if num_lines:
        lines = islice(input_path.open("r"), num_lines)
        text = "\n".join(lines)
    else:
        text = input_file.read()

    # Encode the text
    start = time.perf_counter()
    token_ids = np.array(tokenizer.encode(text), dtype=np.uint16)
    duration = time.perf_counter() - start

    # Write out compression ratio and throughput
    typer.echo(f"Encoded text into {len(token_ids)} tokens")

    compression_ratio = len(text.encode()) / len(token_ids)
    throughput = len(text.encode()) / duration

    typer.echo(f"Tokenizer compression ratio: {compression_ratio:.2f}")
    typer.echo(f"Tokenizer throughput: {throughput:.2f}")

    token_ids_path = input_path.with_suffix(".tokens.npy")
    np.save(token_ids_path, token_ids)

    typer.echo(f"Serialized token ids to {token_ids_path}")


@app.command()
def train_model(
    dataset: Literal["tinystories", "owt"] = "tinystories",
    batch_size: int = 32,
    context_length: int = 256,
    device: str = "mps:0",
    d_model: int = 512,
    num_heads: int = 16,
    d_ff: int = 1344,
    vocab_size: int = 10000,
    num_layers: int = 4,
    rope_theta: float = 10000,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    num_steps: int = 5000,
    checkpoint_path: Path = ROOT_DIRECTORY_PATH.joinpath("checkpoints"),
):
    model = Transformer(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        vocab_size=vocab_size,
        context_length=context_length,
        num_layers=num_layers,
        rope_theta=rope_theta,
        device=torch.device(device),
    )
    model = torch.compile(model, backend="aot_eager")
    optimizer = AdamW(
        params=model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
    )

    model_name = f"{dataset}.pt"
    wandb_run = wandb.init(
        project="cs336-assignment-1",
        config={
            "dataset": dataset,
            "lr": lr,
        },
    )

    dataset_path = ROOT_DIRECTORY_PATH.joinpath("data", dataset)

    train_dataset = np.load(dataset_path.joinpath(f"{dataset}-train-tokens.npy"), mmap_mode="r")
    valid_dataset = np.load(dataset_path.joinpath(f"{dataset}-valid-tokens.npy"), mmap_mode="r")
    for step in range(num_steps):
        # Grab some examples from the training set
        train_inputs, train_targets = load_data(train_dataset, batch_size, context_length, device)

        # Update the learning rate dynamically
        current_lr = cosine_lr_schedule(
            t=step,
            lr_max=10 * lr,
            lr_min=lr,
            t_w=int(0.1 * num_steps),
            t_c=num_steps,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        # Zero out the gradients
        optimizer.zero_grad()

        # Use the current model to generate the logits
        train_outputs = model(train_inputs)

        # Calculate the loss, then compute the gradients with respect to the loss
        train_loss = cross_entropy(train_outputs, train_targets)
        train_loss.backward()

        # Evaluate the model on the validation set
        if step % 20 == 0:
            model.eval()

            valid_inputs, valid_targets = load_data(valid_dataset, batch_size, context_length, device)
            with torch.no_grad():
                valid_outputs = model(valid_inputs)
                valid_loss = cross_entropy(valid_outputs, valid_targets)

            data = {
                "train_loss": train_loss.item(),
                "valid_loss": valid_loss.item(),
            }
            typer.echo(f"{step=}, {data=}")
            wandb_run.log(data, step=step)

            model.train()

        # With our gradients, then update our parameters
        clip_grad_norm(model.parameters(), max_norm=1.0)
        optimizer.step()

    typer.echo("Done training.")
    save_checkpoint(model, optimizer, step, out=checkpoint_path.joinpath(model_name))


@app.command()
def generate(
    prompt: str,
    checkpoint_path: Path,
    dataset: Literal["tinystories", "owt"] = "tinystories",
    context_length: int = 256,
    d_model: int = 512,
    num_heads: int = 16,
    d_ff: int = 1344,
    vocab_size: int = 10000,
    num_layers: int = 4,
    rope_theta: float = 10000,
    device: str = "mps:0",
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
):
    vocab_path = ROOT_DIRECTORY_PATH.joinpath("data", dataset, f"{dataset}-train-vocab.pkl")
    merges_path = ROOT_DIRECTORY_PATH.joinpath("data", dataset, f"{dataset}-train-merges.pkl")
    tokenizer = BPETokenizer.from_files(vocab_path, merges_path, ["<|endoftext|>"])

    model = Transformer(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        vocab_size=vocab_size,
        context_length=context_length,
        num_layers=num_layers,
        rope_theta=rope_theta,
        device=torch.device(device),
    )
    model = torch.compile(model, backend="aot_eager")

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model"])

    tokens = torch.tensor(tokenizer.encode(prompt), device=device)
    iteration = 0
    while True:
        if iteration == max_tokens:
            break

        # Sample the model, and retrieve the probability distribution for the next token
        logits = model(tokens)
        logits = torch.select(logits, dim=-2, index=-1)

        # With temperature sampling, we scale the logits by a parameter T.
        # Intuitively, a lower T causes the scaled distribution to become more peaky,
        # while a higher T causes the scaled distribution to become more uniform.
        if temperature:
            logits = logits / temperature

        probs = softmax(logits, dim=-1)

        # With top-p sampling, we sample from the cdf of the most probable tokens
        # (which is also known as the probably "nucleus")
        if top_p:
            # First, sort the probabilities from smallest to greatest
            sorted_indices = torch.argsort(probs, dim=-1)
            sorted_probs = torch.take_along_dim(probs, sorted_indices, dim=-1)

            # Calculate the cumulative probabilities for the sorted probabilities
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            inverse_indices = torch.scatter(
                torch.zeros_like(sorted_indices),
                dim=-1,
                index=sorted_indices,
                src=torch.arange(sorted_indices.shape[-1]),
            )
            cumulative_probs = torch.take_along_dim(cumulative_probs, inverse_indices, dim=-1)

            # Don't sample token probabilities outside of the cdf threshold
            probs = torch.where(cumulative_probs > 1 - top_p, probs, 0)

        # Sample the next token
        next_token = torch.distributions.Categorical(probs).sample()
        next_token_id = next_token.item()
        tokens = torch.concat([tokens, next_token.unsqueeze(0)])

        assert isinstance(next_token_id, int)
        decoded_token = tokenizer.decode([next_token_id])
        if decoded_token in tokenizer.special_tokens:
            break

        typer.echo(decoded_token, nl=False)

        iteration += 1


if __name__ == "__main__":
    app()
