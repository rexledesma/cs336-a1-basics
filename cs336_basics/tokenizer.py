import os
import pickle
from collections.abc import Iterable, Iterator
from itertools import chain, pairwise
from pathlib import Path

from tqdm import tqdm

from cs336_basics.train_bpe import merge, pre_tokenize


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
    ):
        return cls(
            pickle.load(Path(vocab_filepath).open("rb")),
            pickle.load(Path(merges_filepath).open("rb")),
            special_tokens,
        )

    def encode(self, text: str | bytes) -> list[int]:
        encoded_text = text.encode() if isinstance(text, str) else text
        pre_tokens = pre_tokenize(encoded_text, self.special_tokens)
        token_ids = list(chain.from_iterable(self.tokenize(pre_token) for pre_token in pre_tokens))

        return token_ids

    def tokenize(self, pre_token: tuple[bytes, ...]) -> list[int]:
        # Deconstruct the pre token into its constituent tokens and return their IDs.
        while True:
            merge_ids: set[tuple[int, tuple[bytes, bytes]]] = set()
            for pair in pairwise(pre_token):
                merge_id = self.id_for_merge.get(pair)

                if merge_id is not None:
                    merge_ids.add((merge_id, pair))

            # If no pairs are found, break the loop
            if not merge_ids:
                break

            # Apply the merge in the same order of creation
            _, merged_pair = min(merge_ids)

            pre_token = merge(pre_token, merged_pair)

        token_ids = [self.id_for_token[i] for i in pre_token]

        return token_ids

    def encode_iterable(self, iterable: Iterable[str | bytes]) -> Iterator[int]:
        for text in tqdm(iterable, desc="encode"):
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        bytes_list = (self.token_for_id[token_id] for token_id in ids)
        text = b"".join(bytes_list).decode(errors="replace")

        return text
