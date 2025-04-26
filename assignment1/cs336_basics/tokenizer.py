import os
from pathlib import Path

import regex

GPT2_TOKENIZER_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def generate_positions_to_merge(pre_token: tuple[int, ...], most_common_pair: tuple[int, int]) -> list[int]:
    return [position for position, pair in enumerate(zip(pre_token, pre_token[1:])) if pair == most_common_pair]


def merge(
    frequencies_for_pre_token: dict[tuple[int, ...], int],
    frequencies_for_index_pairs: dict[tuple[int, int], int],
    most_common_pair: tuple[int, int],
    new_index: int,
) -> tuple[
    dict[tuple[int, ...], int],
    dict[tuple[int, int], int],
]:
    new_frequencies_for_pre_token = {}

    for pre_token, frequency in frequencies_for_pre_token.items():
        new_pre_token = []
        positions_to_merge = set(generate_positions_to_merge(pre_token, most_common_pair))

        jdx = 0
        while jdx < len(pre_token):
            if jdx in positions_to_merge:
                # Decrement frequencies for pairs being replaced
                if jdx > 0:
                    prev_pair = (pre_token[jdx - 1], pre_token[jdx])
                    frequencies_for_index_pairs[prev_pair] -= frequency
                if jdx < len(pre_token) - 2:
                    next_pair = (pre_token[jdx + 1], pre_token[jdx + 2])
                    frequencies_for_index_pairs[next_pair] -= frequency

                # Add the merged token
                new_pre_token.append(new_index)
                jdx += 2

                # Increment frequencies for new pairs
                if len(new_pre_token) > 1:
                    new_pair = new_pre_token[-2], new_pre_token[-1]
                    frequencies_for_index_pairs[new_pair] = frequencies_for_index_pairs.get(new_pair, 0) + frequency

                if jdx < len(pre_token):
                    new_pair = new_pre_token[-1], pre_token[jdx]
                    frequencies_for_index_pairs[new_pair] = frequencies_for_index_pairs.get(new_pair, 0) + frequency
            else:
                new_pre_token.append(pre_token[jdx])

                jdx += 1

        new_frequencies_for_pre_token[tuple(new_pre_token)] = frequency

    frequencies_for_index_pairs.pop(most_common_pair, None)

    return new_frequencies_for_pre_token, frequencies_for_index_pairs


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

    index_pair_merges: list[tuple[bytes, bytes]] = []
    vocabulary_for_index: dict[int, bytes] = {x: bytes([x]) for x in range(256)}

    special_token_pattern = "|".join(special_tokens)
    split_corpus = regex.split(special_token_pattern, Path(input_path).read_text())

    # Count the number of occurences for each pair of tokens
    frequencies_for_pre_token: dict[tuple[int, ...], int] = {}
    for corpus in split_corpus:
        for pre_token_match in regex.finditer(GPT2_TOKENIZER_PATTERN, corpus):
            pre_token = tuple(pre_token_match.group(0).encode())

            frequencies_for_pre_token[pre_token] = frequencies_for_pre_token.setdefault(pre_token, 0) + 1

    # Get the frequency of pairs
    frequencies_for_index_pairs: dict[tuple[int, int], int] = {}
    for pre_token, frequency in frequencies_for_pre_token.items():
        for pair in zip(pre_token, pre_token[1:]):
            frequencies_for_index_pairs[pair] = frequencies_for_index_pairs.setdefault(pair, 0) + frequency

    # Add special tokens to the vocabulary
    num_merges = vocab_size - len(vocabulary_for_index)
    for _ in range(num_merges):
        # Find most common pair
        most_common_index_pair = max(frequencies_for_index_pairs, key=lambda p: (frequencies_for_index_pairs[p], p))
        index_1, index_2 = most_common_index_pair

        # Merge the pair
        new_index = len(vocabulary_for_index)
        index_pair_merges.append((vocabulary_for_index[index_1], vocabulary_for_index[index_2]))
        vocabulary_for_index[new_index] = vocabulary_for_index[index_1] + vocabulary_for_index[index_2]

        frequencies_for_pre_token, frequencies_for_index_pairs = merge(
            frequencies_for_pre_token, frequencies_for_index_pairs, most_common_index_pair, new_index
        )

    return vocabulary_for_index, index_pair_merges
