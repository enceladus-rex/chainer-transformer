from chainer_transformer import util

import pytest


def test_build_vocabulary():
  data = 'a b b c c c d d d d'
  end = None
  vocabulary = util.build_vocabulary(data)
  assert vocabulary == {
    (('a',), (end,)): 1,
    (('b',), (end,)): 2,
    (('c',), (end,)): 3,
    (('d',), (end,)): 4,
  }


def test_extract_pairs():
  word = (('a',), ('b',), ('c',), ('d',), (None,))
  pairs = util.extract_pairs(word)
  assert pairs == [
    (('a',), ('b',)),
    (('b',), ('c',)),
    (('c',), ('d',)),
    (('d',), (None,)),
  ]


def test_binary_pair_encoding():
  vocab = util.build_vocabulary('a b c')
  util.binary_pair_encoding(vocab, 1)