import os
import struct
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import pairwise
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import BinaryIO, Self

import regex
from pqdict import pqdict
from tqdm import tqdm

GPT2_TOKENIZER_PATTERN = rb"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


@dataclass
class PairIndex:
    pair: tuple[bytes, bytes]
    frequency: int
    pre_tokens: set[tuple[bytes, ...]]


def merge(pre_token: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    new_pre_token = []
    i = 0
    while i < len(pre_token):
        if i < len(pre_token) - 1 and (pre_token[i], pre_token[i + 1]) == pair:
            new_pre_token.append(b"".join(pair))
            i += 2
        else:
            new_pre_token.append(pre_token[i])
            i += 1

    return tuple(new_pre_token)


def pre_tokenize(encoded_text: bytes, special_tokens: list[str]) -> Iterator[tuple[bytes, ...]]:
    split_corpus = [encoded_text]
    if special_tokens:
        # Attempt to match the longest special token first.
        # This is to ensure that special tokens that are substrings of other special tokens
        # are not split incorrectly.
        special_tokens = sorted(special_tokens, reverse=True)

        special_token_pattern = "|".join(map(regex.escape, special_tokens))
        special_token_pattern = f"({special_token_pattern})".encode()
        split_corpus = regex.splititer(special_token_pattern, encoded_text)

    encoded_special_tokens = {token.encode() for token in special_tokens}
    for corpus in split_corpus:
        if not corpus:
            continue

        if corpus in encoded_special_tokens:
            yield (corpus,)
            continue

        for pre_token_match in regex.finditer(GPT2_TOKENIZER_PATTERN, corpus):
            pre_token = pre_token_match.group()

            yield struct.unpack("c" * len(pre_token), pre_token)


def build_indexes_for_pair(
    frequencies_for_pre_token: dict[tuple[bytes, ...], int],
) -> dict[tuple[bytes, bytes], PairIndex]:
    indexes_for_pair: dict[tuple[bytes, bytes], PairIndex] = {}
    for pre_token, frequency in tqdm(frequencies_for_pre_token.items(), desc="index"):
        for pair in pairwise(pre_token):
            pair_index = indexes_for_pair.setdefault(pair, PairIndex(pair, 0, set()))

            pair_index.frequency += frequency
            pair_index.pre_tokens.add(pre_token)

    return indexes_for_pair


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

    pre_tokens = pre_tokenize(encoded_text, special_tokens)
    frequencies_for_pre_token = Counter(pre_tokens)

    return frequencies_for_pre_token


class BpeIndex:
    def __init__(self, frequencies_for_pre_token: dict[tuple[bytes, ...], int], index_pqdict: pqdict):
        self.frequencies_for_pre_token = frequencies_for_pre_token
        self.index_pqdict = index_pqdict

    @classmethod
    def build(cls, input_path: str | os.PathLike, special_tokens: list[str]) -> Self:
        chunk_boundaries = find_chunk_boundaries(
            Path(input_path).open("rb"),
            desired_num_chunks=cpu_count(),
            split_special_token=special_tokens[0].encode() if special_tokens else b"<|endoftext|>",
        )
        boundaries = pairwise(chunk_boundaries)

        with Pool(cpu_count()) as pool:
            chunk_results = pool.starmap(
                build_frequencies,
                ((input_path, boundary, special_tokens) for boundary in boundaries),
            )

        frequencies_for_pre_token = sum(tqdm(chunk_results, desc="frequencies"), Counter())
        indexes_for_pair = build_indexes_for_pair(frequencies_for_pre_token)
        index_pqdict = pqdict(indexes_for_pair, key=lambda index: (index.frequency, index.pair), reverse=True)

        return cls(frequencies_for_pre_token, index_pqdict)

    def merge_most_frequent_pair(self) -> tuple[bytes, bytes]:
        most_frequent_pair_index: PairIndex = self.index_pqdict.popvalue()

        self._update_index(most_frequent_pair_index)

        return most_frequent_pair_index.pair

    def _update_index(self, most_frequent_pair_index: PairIndex):
        indexes_to_update: dict[tuple[bytes, bytes], PairIndex] = {}
        for pre_token in most_frequent_pair_index.pre_tokens:
            new_pre_token = merge(pre_token, most_frequent_pair_index.pair)

            frequency = self.frequencies_for_pre_token.pop(pre_token)
            self.frequencies_for_pre_token[new_pre_token] = frequency

            for pair in pairwise(new_pre_token):
                pair_index = indexes_to_update.setdefault(pair, self.index_pqdict.get(pair, PairIndex(pair, 0, set())))

                pair_index.frequency += frequency
                pair_index.pre_tokens.add(new_pre_token)

            for pair in pairwise(pre_token):
                if pair != most_frequent_pair_index.pair:
                    pair_index = indexes_to_update.setdefault(pair, self.index_pqdict[pair])
                    pair_index.frequency -= frequency
                    pair_index.pre_tokens.discard(pre_token)

        for pair, pair_index in indexes_to_update.items():
            if pair not in self.index_pqdict:
                self.index_pqdict.additem(pair, pair_index)
            else:
                self.index_pqdict.updateitem(pair, pair_index)


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[
    dict[int, bytes],
    list[tuple[bytes, bytes]],
]:
    initial_vocabulary = [token.encode() for token in special_tokens] + [x.to_bytes() for x in range(256)]
    vocabulary_for_index: dict[int, bytes] = dict(enumerate(initial_vocabulary))
    index_pair_merges: list[tuple[bytes, bytes]] = []

    bpe_index = BpeIndex.build(input_path, special_tokens)

    initial_vocabulary_size = len(initial_vocabulary)
    for new_vocab_id in tqdm(range(initial_vocabulary_size, vocab_size), desc="vocab"):
        most_frequent_pair = bpe_index.merge_most_frequent_pair()

        index_pair_merges.append(most_frequent_pair)
        vocabulary_for_index[new_vocab_id] = b"".join(most_frequent_pair)

    return vocabulary_for_index, index_pair_merges
