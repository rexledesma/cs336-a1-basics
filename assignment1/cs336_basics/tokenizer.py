import os
import pickle
from collections.abc import Iterable, Iterator
from itertools import chain, pairwise
from pathlib import Path
from typing import Self

from cs336_basics.train_bpe import pre_tokenize


class BPETokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.token_for_id = vocab
        self.id_for_token = {token: token_id for token_id, token in self.token_for_id.items()}
        self.id_for_merge = {merge: merge_id for merge_id, merge in enumerate(merges)}
        self.special_tokens = special_tokens or []

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: list[str] | None = None,
    ) -> Self:
        return cls(
            pickle.load(Path(vocab_filepath).open("rb")),
            pickle.load(Path(merges_filepath).open("rb")),
            special_tokens,
        )

    def encode(self, text: str) -> list[int]:
        pre_tokens = pre_tokenize(text.encode(), self.special_tokens)

        token_ids = list(chain.from_iterable(self.tokenize(pre_token) for pre_token in pre_tokens))

        return token_ids

    def tokenize(self, pre_token: tuple[bytes, ...]) -> list[int]:
        # Deconstruct the pre token into its constituent tokens and return their IDs.
        pre_token_list = list(pre_token)
        while len(pre_token_list) > 1:
            merge_positions: list[tuple[int, int]] = []
            for idx, pair in enumerate(pairwise(pre_token_list)):
                merge_id = self.id_for_merge.get(pair)

                if merge_id is not None:
                    merge_positions.append((merge_id, idx))

            # If no pairs are found, break the loop
            if not merge_positions:
                break

            # Apply the merge in the same order of creation
            merge_id_to_process, _ = min(merge_positions)

            merge_position = min(idx for (merge_id, idx) in merge_positions if merge_id == merge_id_to_process)
            merged_bytes = b"".join((pre_token_list[merge_position], pre_token_list[merge_position + 1]))

            pre_token_list[merge_position] = merged_bytes
            del pre_token_list[merge_position + 1]

        token_ids = [self.id_for_token[i] for i in pre_token_list]

        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        bytes_list = (self.token_for_id[token_id] for token_id in ids)
        text = b"".join(bytes_list).decode(errors="replace")

        return text
