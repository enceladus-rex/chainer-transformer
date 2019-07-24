from chainer_transformer import util

from pprint import pprint

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


def test_binary_pair_encoding_simple():
  vocab = util.build_vocabulary('a b c')
  token_indices, tokens = util.binary_pair_encoding(vocab, 1)
  assert tokens == {
    ('a', None): 1,
    ('b', None): 1,
    ('c', None): 1,
  }


def test_binary_pair_encoding_example():
  vocab = util.build_vocabulary((
    'low low low low low '
    'lower lower '
    'newest newest newest newest newest newest '
    'widest widest widest'
  ))

  assert vocab == {
    (('l',), ('o',), ('w',), (None,)): 5,
    (('l',), ('o',), ('w',), ('e',), ('r',), (None,)): 2,
    (('n',), ('e',), ('w',), ('e',), ('s',), ('t',), (None,)): 6,
    (('w',), ('i',), ('d',), ('e',), ('s',), ('t',), (None,)): 3,
  }

  num_merges = 10
  token_indices, tokens = util.binary_pair_encoding(vocab, num_merges)
  
  print()
  pprint(tokens)
  print()
  pprint(list(zip(range(len(token_indices)), token_indices)))