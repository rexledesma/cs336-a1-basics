import pickle
import time
from itertools import islice
from pathlib import Path

import typer

from cs336_basics.tokenizer import BPETokenizer
from cs336_basics.train_bpe import train_bpe

app = typer.Typer()


@app.command()
def create_bpe(input_path: Path, vocab_size: int):
    """Train BPE on a text file and serialize the vocabulary and merges."""
    start = time.perf_counter()
    vocabulary, merges = train_bpe(input_path, vocab_size, ["<|endoftext|>"])
    duration = time.perf_counter() - start

    typer.echo(f"Vocabulary size: {len(vocabulary)}")
    typer.echo(f"Training took {duration:.2f} seconds")

    filename = input_path.name
    vocab_path = input_path.joinpath("..", f"{filename}_vocab.pkl").resolve()
    merges_path = input_path.joinpath("..", f"{filename}_merges.pkl").resolve()

    # Serialize vocabulary and merges
    vocab_path.unlink(missing_ok=True)
    merges_path.unlink(missing_ok=True)

    pickle.dump(vocabulary, vocab_path.open("wb"))
    pickle.dump(merges, merges_path.open("wb"))

    typer.echo(f"Serialized vocabulary to {vocab_path}")
    typer.echo(f"Serialized merges to {merges_path}")


@app.command()
def run_bpe(vocab_path: Path, merges_path: Path, input_path: Path, num_lines: int | None):
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
    token_ids = tokenizer.encode(text)

    # Write out compression ratio
    typer.echo(f"Encoded text into {len(token_ids)} tokens")

    compression_ratio = len(text.encode()) / len(token_ids)
    typer.echo(f"Tokenizer compression ratio: {compression_ratio:.2f}")


if __name__ == "__main__":
    app()
