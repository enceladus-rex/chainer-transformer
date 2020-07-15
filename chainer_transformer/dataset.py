from chainer.datasets import TextDataset, TransformDataset

import chainer.functions as F

from typing import NamedTuple, Dict, List, Optional

import numpy as np

try:
    import cupy as xp
except ImportError:
    import numpy as xp


def pad_ids(x, value, length):
    assert len(x) <= length
    if len(x) < length:
        return x + [value] * (length - len(x))
    return x


class Vocab(NamedTuple):
    embeddings: xp.array
    id_to_bpe: List[str]
    bpe_to_id: Dict[str, int]

    @property
    def vocab_size(self) -> int:
        return self.embeddings.shape[0]

    @property
    def embedding_size(self) -> int:
        return self.embeddings.shape[-1]

    @property
    def start_id(self) -> int:
        return len(self.id_to_bpe)

    @property
    def end_id(self) -> int:
        return len(self.id_to_bpe) + 1

    def embed(self, ids):
        return F.embed_id(xp.array(ids), self.embeddings)

    def transform(self,
                  line,
                  chunk_length: Optional[int] = None,
                  include_start: bool = False) -> xp.array:
        if line.endswith('\n'):
            line = line[:-1]
        tokens = line.split(sep=' ')
        ids = []
        if include_start:
            ids.append(self.start_id)

        for t in tokens:
            if t not in self.bpe_to_id:
                import pdb
                pdb.set_trace()
            ids.append(self.bpe_to_id[t])

        ids.append(self.end_id)
        token_length = len(ids)
        if chunk_length is not None:
            assert token_length <= chunk_length, 'number of tokens greater than chunk length'
            ids = pad_ids(ids, self.end_id, chunk_length)

        mask = xp.zeros(len(ids), dtype=xp.float32)
        mask[:token_length] = 1.

        return xp.array(ids), mask


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


def make_vocab(glove_filename) -> Vocab:
    with open(glove_filename, 'rb') as f:
        data = np.load(f)
        tokens = data['tokens']
        embeddings = xp.array(data['embeddings'], dtype=xp.float32)

        assert len(embeddings) == (len(tokens) + 2), 'invalid glove data'

        id_to_bpe = tokens
        bpe_to_id = {bpe: i for i, bpe in enumerate(id_to_bpe)}
        return Vocab(embeddings, id_to_bpe, bpe_to_id)


def filter_example(x, y, max_length=None):
    if x.endswith('\n'):
        x = x[:-1]
    if y.endswith('\n'):
        y = y[:-1]
    x_within_length = max_length is None or len(x.split(sep=' ')) <= max_length
    y_within_length = max_length is None or len(y.split(sep=' ')) <= max_length
    return x and y and x_within_length and y_within_length


def make_dataset(source_bpe_filename,
                 target_bpe_filename,
                 source_vocab,
                 target_vocab,
                 chunk_length=1000):

    d = TextDataset(
        (source_bpe_filename, target_bpe_filename),
        filter_func=lambda x, y: filter_example(x, y, chunk_length - 2))
    return TransformDataset(
        d, TokenTransformer(source_vocab, target_vocab, chunk_length))
