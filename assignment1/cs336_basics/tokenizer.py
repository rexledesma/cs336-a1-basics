import os
from pathlib import Path

import regex

GPT2_TOKENIZER_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


# def generate_positions_to_merge(pre_token: tuple[bytes, ...], most_frequent_pair: tuple[bytes, bytes]) -> list[int]:
#     return [position for position, pair in enumerate(zip(pre_token, pre_token[1:])) if pair == most_frequent_pair]


# def merge(
#     frequencies_for_pre_token: dict[tuple[bytes, ...], int],
#     frequencies_for_pair: dict[tuple[bytes, bytes], int],
#     most_frequent_pair: tuple[bytes, bytes],
#     new_index: int,
# ) -> tuple[
#     dict[tuple[bytes, ...], int],
#     dict[tuple[bytes, bytes], int],
# ]:
#     new_frequencies_for_pre_token = {}

#     for pre_token, frequency in frequencies_for_pre_token.items():
#         new_pre_token = []
#         positions_to_merge = set(generate_positions_to_merge(pre_token, most_frequent_pair))

#         jdx = 0
#         while jdx < len(pre_token):
#             if jdx in positions_to_merge:
#                 # Decrement frequencies for pairs being replaced
#                 if jdx > 0:
#                     prev_pair = (pre_token[jdx - 1], pre_token[jdx])
#                     frequencies_for_pair[prev_pair] -= frequency
#                 if jdx < len(pre_token) - 2:
#                     next_pair = (pre_token[jdx + 1], pre_token[jdx + 2])
#                     frequencies_for_pair[next_pair] -= frequency

#                 # Add the merged token
#                 new_pre_token.append(new_index)
#                 jdx += 2

#                 # Increment frequencies for new pairs
#                 if len(new_pre_token) > 1:
#                     new_pair = new_pre_token[-2], new_pre_token[-1]
#                     frequencies_for_pair[new_pair] = frequencies_for_pair.get(new_pair, 0) + frequency

#                 if jdx < len(pre_token):
#                     new_pair = new_pre_token[-1], pre_token[jdx]
#                     frequencies_for_pair[new_pair] = frequencies_for_pair.get(new_pair, 0) + frequency
#             else:
#                 new_pre_token.append(pre_token[jdx])

#                 jdx += 1

#         new_frequencies_for_pre_token[tuple(new_pre_token)] = frequency

#     frequencies_for_pair.pop(most_frequent_pair, None)

#     return new_frequencies_for_pre_token, frequencies_for_pair


def merge(
    frequencies_for_pre_token: dict[tuple[bytes, ...], int],
    most_frequent_pair: tuple[bytes, bytes],
) -> dict[tuple[bytes, ...], int]:
    new_frequencies_for_pre_token = {}

    for pre_token, frequency in frequencies_for_pre_token.items():
        new_pre_token = []
        idx = 0
        while idx < len(pre_token):
            if idx < len(pre_token) - 1 and (pre_token[idx], pre_token[idx + 1]) == most_frequent_pair:
                # Merge the pair
                new_pre_token.append(b"".join(most_frequent_pair))
                idx += 2
            else:
                new_pre_token.append(pre_token[idx])
                idx += 1

        new_pre_token = tuple(new_pre_token)
        new_frequencies_for_pre_token[new_pre_token] = new_frequencies_for_pre_token.get(new_pre_token, 0) + frequency

    return new_frequencies_for_pre_token


def pre_tokenize(input_path: str | os.PathLike, special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
    """
    Pre-tokenize the input text using the GPT-2 tokenizer pattern.

    Args:
        input_path (Path): Path to a text file with BPE tokenizer training data.
    Returns:
        dict[tuple[bytes, ...], int]: A dictionary mapping pre-tokenized text to its frequency.
    """
    special_token_pattern = "|".join(map(regex.escape, special_tokens))
    split_corpus = regex.split(special_token_pattern, Path(input_path).read_text())

    frequencies_for_pre_token: dict[tuple[bytes, ...], int] = {}
    for corpus in split_corpus:
        for pre_token_match in regex.finditer(GPT2_TOKENIZER_PATTERN, corpus):
            pre_token = tuple(char.encode() for char in pre_token_match.group())

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


def train_bpe_tokenizer(
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

    # Count the number of occurences for each pair of tokens
    frequencies_for_pre_token = pre_tokenize(input_path, special_tokens)

    # Add special tokens to the vocabulary
    initial_vocabulary_size = len(initial_vocabulary)
    for new_vocab_id in range(initial_vocabulary_size, vocab_size):
        # Get the frequency of pairs
        frequencies_for_pairs = build_pair_frequencies(frequencies_for_pre_token)

        # Find most common pair
        most_frequent_pair = max(frequencies_for_pairs, key=lambda p: (frequencies_for_pairs[p], p))

        # Merge the pair
        index_pair_merges.append(most_frequent_pair)
        vocabulary_for_index[new_vocab_id] = b"".join(most_frequent_pair)

        frequencies_for_pre_token = merge(frequencies_for_pre_token, most_frequent_pair)

    return vocabulary_for_index, index_pair_merges
