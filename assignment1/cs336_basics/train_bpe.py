import os
from collections.abc import Iterator
from pathlib import Path

import regex

GPT2_TOKENIZER_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


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


def pre_tokenize(
    input_path: str | os.PathLike, special_tokens: list[str], *, keep_special_tokens: bool
) -> Iterator[tuple[bytes, ...]]:
    """
    Pre-tokenize the input text using the GPT-2 tokenizer pattern.

    Args:
        input_path (Path): Path to a text file with BPE tokenizer training data.
        special_tokens (list[str]): A list of strings to add to the vocabulary.
        keep_special_tokens (bool): If True, keep special tokens in the output.
    Returns:
        Iterator[tuple[bytes, ...]]: An iterator over pre-tokenized text.
    """

    contents = Path(input_path).read_text() if isinstance(input_path, os.PathLike) else input_path

    special_token_pattern = "|".join(map(regex.escape, special_tokens))
    if keep_special_tokens:
        special_token_pattern = f"({special_token_pattern})"

    split_corpus = regex.split(special_token_pattern, contents)

    for corpus in split_corpus:
        if keep_special_tokens and regex.match(special_token_pattern, corpus):
            # If the corpus is a special token, yield it as a single pre-token
            pre_token = tuple(bytes([i]) for i in corpus.encode())

            yield pre_token
        else:
            for pre_token_match in regex.finditer(GPT2_TOKENIZER_PATTERN, corpus):
                pre_token = tuple(bytes([i]) for i in pre_token_match.group().encode())

                yield pre_token


def build_frequencies_for_pre_token(
    pre_tokens: Iterator[tuple[bytes, ...]],
) -> dict[tuple[bytes, ...], int]:
    """
    Build a dictionary of frequencies for each pre-tokenized text.

    Args:
        pre_tokens (Iterator[tuple[bytes, ...]]): An iterator over pre-tokenized text.
    Returns:
        dict[tuple[bytes, ...], int]: A dictionary mapping pre-tokenized text to its frequency.
    """
    frequencies_for_pre_token: dict[tuple[bytes, ...], int] = {}
    for pre_token in pre_tokens:
        frequencies_for_pre_token[pre_token] = frequencies_for_pre_token.setdefault(pre_token, 0) + 1

    return frequencies_for_pre_token


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
    # TODO: parallelize pre-tokenization

    initial_vocabulary = [token.encode() for token in special_tokens] + [bytes([x]) for x in range(256)]
    vocabulary_for_index: dict[int, bytes] = dict(enumerate(initial_vocabulary))
    index_pair_merges: list[tuple[bytes, bytes]] = []

    # Pre-tokenize the input text
    pre_tokens = pre_tokenize(input_path, special_tokens, keep_special_tokens=False)

    # Count the number of occurences for each pair of tokens
    frequencies_for_pre_token = build_frequencies_for_pre_token(pre_tokens)

    # Get the frequency of pairs
    frequencies_for_pairs = build_pair_frequencies(frequencies_for_pre_token)

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
