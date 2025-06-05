import os
import struct
from collections import Counter
from collections.abc import Iterator
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import BinaryIO

import regex

GPT2_TOKENIZER_PATTERN = rb"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def generate_positions_to_merge(pre_token: tuple[bytes, ...], most_frequent_pair: tuple[bytes, bytes]) -> list[int]:
    return [position for position, pair in enumerate(zip(pre_token, pre_token[1:])) if pair == most_frequent_pair]


def merge(
    frequencies_for_pre_token: dict[tuple[bytes, ...], int],
    frequencies_for_pair: dict[tuple[bytes, bytes], int],
    most_frequent_pair: tuple[bytes, bytes],
) -> tuple[
    dict[tuple[bytes, ...], int],
    dict[tuple[bytes, bytes], int],
]:
    new_frequencies_for_pre_token = {}

    for pre_token, frequency in frequencies_for_pre_token.items():
        new_pre_token = []
        positions_to_merge = set(generate_positions_to_merge(pre_token, most_frequent_pair))

        idx = 0
        while idx < len(pre_token):
            if idx in positions_to_merge:
                # Decrement frequencies for pairs being replaced
                if idx > 0:
                    prev_pair = (pre_token[idx - 1], pre_token[idx])
                    frequencies_for_pair[prev_pair] -= frequency
                if idx < len(pre_token) - 2:
                    next_pair = (pre_token[idx + 1], pre_token[idx + 2])
                    frequencies_for_pair[next_pair] -= frequency

                # Add the merged token
                new_pre_token.append(b"".join(most_frequent_pair))
                idx += 2

                # Increment frequencies for new pairs
                if len(new_pre_token) > 1:
                    new_pair = new_pre_token[-2], new_pre_token[-1]
                    frequencies_for_pair[new_pair] = frequencies_for_pair.get(new_pair, 0) + frequency

                if idx < len(pre_token):
                    new_pair = new_pre_token[-1], pre_token[idx]
                    frequencies_for_pair[new_pair] = frequencies_for_pair.get(new_pair, 0) + frequency
            else:
                new_pre_token.append(pre_token[idx])

                idx += 1

        new_frequencies_for_pre_token[tuple(new_pre_token)] = frequency

    frequencies_for_pair.pop(most_frequent_pair)

    return new_frequencies_for_pre_token, frequencies_for_pair


def pre_tokenize(encoded_text: bytes, special_tokens: list[str]) -> Iterator[tuple[bytes, ...]]:
    """
    Pre-tokenize the input text using the GPT-2 tokenizer pattern.

    Args:
        encoded_text (bytes): The BPE tokenizer training data, represented in bytes.
        special_tokens (list[str]): A list of strings to add to the vocabulary.
    Returns:
        Iterator[bytes]: An iterator over pre-tokenized text.
    """

    split_corpus = [encoded_text]
    if special_tokens:
        special_token_pattern = "|".join(map(regex.escape, special_tokens)).encode()
        split_corpus = regex.splititer(special_token_pattern, encoded_text)

    for corpus in split_corpus:
        for pre_token_match in regex.finditer(GPT2_TOKENIZER_PATTERN, corpus):
            pre_token = pre_token_match.group()

            yield struct.unpack("c" * len(pre_token), pre_token)


def build_pair_frequencies(frequencies_for_pre_token: dict[tuple[bytes, ...], int]) -> dict[tuple[bytes, bytes], int]:
    """
    Build a dictionary of frequencies for each pair of tokens in the pre-tokenized text.

    Args:
        frequencies_for_pre_token (dict[tuple[int, ...], int]): A dictionary mapping
            pre-tokenized text to its frequency.
    Returns:
        dict[tuple[bytes, bytes], int]: A dictionary mapping pairs of tokens to their frequency.
    """

    frequencies_for_pair: dict[tuple[bytes, bytes], int] = {}
    for pre_token, frequency in frequencies_for_pre_token.items():
        for pair in zip(pre_token, pre_token[1:]):
            frequencies_for_pair[pair] = frequencies_for_pair.setdefault(pair, 0) + frequency

    return frequencies_for_pair


def find_chunk_boundaries(file: BinaryIO, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def build_frequencies(input_path: str | os.PathLike, boundary: tuple[int, int], special_tokens: list[str]):
    # Read contents from file
    file = Path(input_path).open("rb")
    file.seek(boundary[0])
    encoded_text = file.read(boundary[1] - boundary[0])

    # Pre-tokenize the input text
    pre_tokens = pre_tokenize(encoded_text, special_tokens)

    # Count the number of occurences for each pair of tokens
    frequencies_for_pre_token = Counter(pre_tokens)

    # Get the frequency of pairs
    frequencies_for_pairs = build_pair_frequencies(frequencies_for_pre_token)

    return frequencies_for_pre_token, frequencies_for_pairs


def build_frequencies_in_parallel(input_path: str | os.PathLike, special_tokens: list[str]):
    boundaries = [(0, Path(input_path).stat().st_size)]
    if os.getenv("CS336_BPE_ENABLE_MULTIPROCESSING") == "1":
        print("Using multiprocessing for BPE training.")
        chunk_boundaries = find_chunk_boundaries(
            Path(input_path).open("rb"),
            desired_num_chunks=cpu_count(),
            split_special_token=special_tokens[0].encode() if special_tokens else b"<|endoftext|>",
        )

        boundaries = list(zip(chunk_boundaries, chunk_boundaries[1:]))

    frequencies_for_pre_token: dict[tuple[bytes, ...], int] = Counter()
    frequencies_for_pairs: dict[tuple[bytes, bytes], int] = Counter()
    with Pool(cpu_count()) as pool:
        for chunked_frequencies_for_pre_token, chunked_frequencies_for_pairs in pool.starmap(
            build_frequencies,
            ((input_path, boundary, special_tokens) for boundary in boundaries),
        ):
            frequencies_for_pre_token.update(chunked_frequencies_for_pre_token)
            frequencies_for_pairs.update(chunked_frequencies_for_pairs)

    return frequencies_for_pre_token, frequencies_for_pairs


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[
    dict[int, bytes],
    list[tuple[bytes, bytes]],
]:
    """
    Train a BPE tokenizer on the input text file.

    Args:
        input_path (Path): Path to a text file with BPE tokenizer training data.
        vocab_size (int): A positive integer that defines the maximum final vocabulary size
            (including the initial byte vocabulary, vocabulary items produced from merging,
            and any special tokens).
        special_tokens (list[str]): A list of strings to add to the vocabulary.
            These special tokens do not otherwise affect BPE training.
    Returns:
        vocab_size (dict[int, bytes]): The tokenizer vocabulary, a mapping from int
            (token ID in the vocabulary) to bytes (token bytes).
        merges (list[tuple[bytes, bytes]]): A list of BPE merges produced from training.
            Each list item is a tuple of bytes (<token1>, <token2>), representing that
            <token1> was merged with <token2>. The merges should be ordered by order of creation.
    """
    initial_vocabulary = [token.encode() for token in special_tokens] + [x.to_bytes() for x in range(256)]
    vocabulary_for_index: dict[int, bytes] = dict(enumerate(initial_vocabulary))
    index_pair_merges: list[tuple[bytes, bytes]] = []

    frequencies_for_pre_token, frequencies_for_pairs = build_frequencies_in_parallel(input_path, special_tokens)

    # Add special tokens to the vocabulary
    initial_vocabulary_size = len(initial_vocabulary)
    for new_vocab_id in range(initial_vocabulary_size, vocab_size):
        # Find most common pair
        most_frequent_pair = max(frequencies_for_pairs, key=lambda p: (frequencies_for_pairs[p], p))

        # Merge the pair
        index_pair_merges.append(most_frequent_pair)
        vocabulary_for_index[new_vocab_id] = b"".join(most_frequent_pair)

        frequencies_for_pre_token, frequencies_for_pairs = merge(
            frequencies_for_pre_token, frequencies_for_pairs, most_frequent_pair
        )

    return vocabulary_for_index, index_pair_merges
