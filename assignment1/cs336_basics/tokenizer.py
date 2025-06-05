import itertools
import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from cs336_basics.train_bpe import pre_tokenize


class BPETokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        """
        Construct a tokenizer from a given vocabulary, list of merges, and (optionally) a list of special tokens.
        """
        self.token_for_id = vocab
        self.id_for_token = {token: token_id for token_id, token in self.token_for_id.items()}
        self.id_for_merge = {merge: merge_id for merge_id, merge in enumerate(merges)}
        self.special_tokens = special_tokens or []

    @classmethod
    def from_files(
        cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None
    ) -> "BPETokenizer":
        """
        Constructs and return a Tokenizer from a serialized vocabulary and list of merges
        (in the same format that your BPE training code output) and (optionally) a list of special tokens.
        """
        vocab = json.loads(
            Path(vocab_filepath).read_text(),
            object_hook=lambda d: {int(k): bytes(v) for k, v in d.items()},
        )
        merges: list[tuple[bytes, bytes]] = []
        for line in Path(merges_filepath).read_text().splitlines():
            x, y = line.split()
            merges.append((x.encode(), y.encode()))

        return cls(
            vocab=vocab,
            merges=merges,
            special_tokens=special_tokens,
        )

    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs.
        """
        pre_tokens = pre_tokenize(text.encode(), self.special_tokens)

        token_ids = list(itertools.chain.from_iterable(self.tokenize(pre_token) for pre_token in pre_tokens))

        return token_ids

    def tokenize(self, pre_token: tuple[bytes, ...]) -> list[int]:
        """
        Tokenize a pre-tokenized input into a sequence of token IDs.
        """
        # Deconstruct the pre token into its constituent tokens and return their IDs.
        pre_token_list = list(pre_token)
        while len(pre_token_list) > 1:
            merge_positions: list[tuple[int, int]] = []
            for idx, pair in enumerate(zip(pre_token_list, pre_token_list[1:])):
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
        """
        Given an iterable of strings (e.g., a Python file handle),
        return a generator that lazily yields token IDs.
        """
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text.
        """
        bytes_list = (self.token_for_id[token_id] for token_id in ids)
        text = b"".join(bytes_list).decode(errors="replace")

        return text
