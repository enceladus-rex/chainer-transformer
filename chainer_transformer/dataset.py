from chainer.datasets import TextDataset, TransformDataset
from chainer_transformer.util import word_token_indices, make_words

try:
  import cupy as xp
except ImportError:
  import numpy as xp


def one_hot_encode(indices, dim):
  return xp.eye(dim)[indices]


def pad_indices(x, value, length):
  assert len(x) <= length
  if len(x) < length:
    return x + [value] * (length - len(x))
  return x


def combine_words(words):
  return sum(words, ())


def filter_data(source_data, target_data, token_trie):
  source_indices = word_token_indices(combine_words(make_words(source_data)),
                                      token_trie)
  target_indices = word_token_indices(combine_words(make_words(target_data)),
                                      token_trie)
  return source_indices is not None and target_indices is not None


class TokenTransformer:
  def __init__(self, token_trie, dim, eol_index, chunk_length=100):
    self.token_trie = token_trie
    self.dim = dim
    self.eol_index = eol_index
    self.chunk_length = chunk_length

  def __call__(self, in_data):
    source_data, target_data = in_data

    source_indices = pad_indices(
        word_token_indices(combine_words(make_words(source_data)),
                           self.token_trie), self.eol_index, self.chunk_length)

    target_indices = pad_indices(
        word_token_indices(combine_words(make_words(target_data)),
                           self.token_trie), self.eol_index, self.chunk_length)

    return (one_hot_encode(source_indices,
                           self.dim), one_hot_encode(target_indices, self.dim))


def make_dataset(source_text, target_text, token_trie, dim, eol_index,
                 chunk_length):
  d = TextDataset((source_text, target_text))
  return TransformDataset(
      d, TokenTransformer(token_trie, dim, eol_index, chunk_length))
