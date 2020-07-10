from chainer.datasets import TextDataset, TransformDataset

from typing import NamedTuple, Dict, List, Optional

try:
    import cupy as xp
except ImportError:
    import numpy as xp


def one_hot_encode(indices, dim):
    encoded = xp.zeros((len(indices), dim), dtype=xp.float32)
    for i, ix in enumerate(indices):
        encoded[i, ix] = 1.
    return encoded


def pad_indices(x, value, length):
    assert len(x) <= length
    if len(x) < length:
        return x + [value] * (length - len(x))
    return x


class Vocab(NamedTuple):
    index_to_bpe: List[str]
    bpe_to_index: Dict[str, int]

    @property
    def vocab_size(self) -> int:
        # Add 1 for EOL indicator.
        return len(self.index_to_bpe) + 1

    @property
    def eol_index(self) -> int:
        return len(self.index_to_bpe)

    def transform(self, line, chunk_length: Optional[int] = None) -> xp.array:
        tokens = line.split()
        indices = [self.bpe_to_index[t] for t in tokens]
        if chunk_length is not None:
            indices = pad_indices(indices, self.eol_index, chunk_length)

        return one_hot_encode(indices, self.vocab_size)


class TokenTransformer:
    def __init__(self, source_bpe_vocab, target_bpe_vocab, chunk_length=1000):
        self.source_bpe_vocab = source_bpe_vocab
        self.target_bpe_vocab = target_bpe_vocab
        self.chunk_length = chunk_length

    def __call__(self, in_data):
        source_data, target_data = in_data
        return self.source_bpe_vocab.transform(
            source_data, self.chunk_length), self.target_bpe_vocab.transform(
                target_data, self.chunk_length)


def make_vocab(raw_bpe_vocab_filename) -> Vocab:
    with open(raw_bpe_vocab_filename, 'r') as f:
        index_to_bpe = [l.split()[0] for l in f.readlines()]
        bpe_to_index = {bpe: i for i, bpe in enumerate(index_to_bpe)}
        return Vocab(index_to_bpe, bpe_to_index)


def make_dataset(source_bpe_filename,
                 target_bpe_filename,
                 source_bpe_vocab_filename,
                 target_bpe_vocab_filename,
                 chunk_length=1000):
    d = TextDataset((source_bpe_filename, target_bpe_filename),
                    filter_func=lambda x, y: x and y)
    return TransformDataset(
        d,
        TokenTransformer(make_vocab(source_bpe_vocab_filename),
                         make_vocab(target_bpe_vocab_filename), chunk_length))
