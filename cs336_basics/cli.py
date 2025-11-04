import pickle
import time
from itertools import islice
from pathlib import Path

import numpy as np
import torch
import typer

from cs336_basics.optimizer import AdamW, cross_entropy
from cs336_basics.tokenizer import BPETokenizer
from cs336_basics.train import load_data, save_checkpoint
from cs336_basics.train_bpe import train_bpe
from cs336_basics.transformer import Transformer, softmax

app = typer.Typer()


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
    dataset_path: Path,
    batch_size: int,
    context_length: int,
    device: str,
    d_model: int,
    num_heads: int,
    d_ff: int,
    vocab_size: int,
    num_layers: int,
    rope_theta: float,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float],
    eps: float,
    iterations: int,
    checkpoint_path: Path,
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
    optimizer = AdamW(
        params=model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
    )

    dataset = np.load(dataset_path, mmap_mode="r")
    inputs, targets = load_data(dataset, batch_size, context_length, device)
    for idx in range(iterations):
        # First, zero out the gradients
        optimizer.zero_grad()

        # Use the current model to generate the logits
        outputs = model(inputs)

        # Calculate the loss, then compute the gradients with respect to the loss
        loss = cross_entropy(outputs, targets)
        loss.backward()

        typer.echo(f"Iteration {idx}: loss={loss.item()}")

        # With our gradients, then update our parameters
        optimizer.step()

    typer.echo("Done training.")
    save_checkpoint(model, optimizer, idx, out=checkpoint_path)


@app.command()
def generate(
    checkpoint_path: Path,
    vocab_path: Path,
    merges_path: Path,
    prompt: str,
    context_length: int,
    d_model: int,
    num_heads: int,
    d_ff: int,
    vocab_size: int,
    num_layers: int,
    rope_theta: float,
    device: str,
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
):
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
