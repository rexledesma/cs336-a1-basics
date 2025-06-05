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
        self.merged_token_for_pair = {
            (
                self.id_for_token[token_1],
                self.id_for_token[token_2],
            ): merge_id
            for merge_id, (token_1, token_2) in enumerate(merges)
        }
        self.special_tokens = special_tokens or []
        self.special_tokens.sort(reverse=True)

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
        pre_tokens = pre_tokenize(text.encode(), self.special_tokens, keep_special_tokens=True)

        token_ids = list(itertools.chain.from_iterable(self.tokenize(pre_token) for pre_token in pre_tokens))

        return token_ids

    def tokenize(self, pre_token: bytes) -> list[int]:
        """
        Tokenize a pre-tokenized input into a sequence of token IDs.
        """
        # If the pre-token in its entirety is in our vocabulary, return its ID directly.
        if token_id := self.id_for_token.get(pre_token):
            return [token_id]

        # Otherwise, deconstruct the pre token into its constituent tokens and return their IDs.
        token_ids = [self.id_for_token[bytes([byte])] for byte in pre_token]
        while len(token_ids) > 1:
            merge_positions: list[tuple[int, int]] = []
            for idx, pair in enumerate(zip(token_ids, token_ids[1:])):
                merge_id = self.merged_token_for_pair.get(pair)

                if merge_id:
                    merge_positions.append((merge_id, idx))

            # If no pairs are found, break the loop
            if not merge_positions:
                break

            # Apply the merge in the same order of creation
            merge_id_to_process, _ = min(merge_positions)

            merge_positions_to_process = {idx for (merge_id, idx) in merge_positions if merge_id == merge_id_to_process}

            new_token_ids = []
            idx = 0
            while idx < len(token_ids):
                if idx in merge_positions_to_process:
                    merged_token = b"".join(
                        (
                            self.token_for_id[token_ids[idx]],
                            self.token_for_id[token_ids[idx + 1]],
                        )
                    )
                    new_token_ids.append(self.id_for_token[merged_token])

                    idx += 2
                else:
                    new_token_ids.append(token_ids[idx])

                    idx += 1

            token_ids = new_token_ids

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
