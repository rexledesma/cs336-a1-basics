import pickle
import time
from pathlib import Path

import typer

from cs336_basics.train_bpe import train_bpe

app = typer.Typer()


@app.command()
def main(input_path: Path, vocab_size: int):
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


if __name__ == "__main__":
    app()
