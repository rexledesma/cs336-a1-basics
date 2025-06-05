import pathlib
import time

from cs336_basics.train_bpe import train_bpe

TINY_STORIES_PATH = pathlib.Path(__file__).joinpath("..", "..", "data", "TinyStoriesV2-GPT4-train.txt").resolve()
MAX_VOCAB_SIZE = 10_000

if __name__ == "__main__":
    start = time.perf_counter()
    vocabulary, merges = train_bpe(TINY_STORIES_PATH, MAX_VOCAB_SIZE, ["<|endoftext|>"])
    duration = time.perf_counter() - start

    print(f"Vocabulary size: {len(vocabulary)}")
    print(f"Merges tail: {merges[-10:]}")
    print(f"Training took {duration:.2f} seconds")
