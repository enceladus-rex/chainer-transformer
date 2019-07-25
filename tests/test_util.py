from chainer_transformer import util

from pprint import pprint

import pytest


def test_build_vocabulary():
  data = 'a b b c c c d d d d'
  vocabulary = util.build_vocabulary(data)
  assert vocabulary == {
      (('a', ), (None, )): 1,
      (('b', ), (None, )): 2,
      (('c', ), (None, )): 3,
      (('d', ), (None, )): 4,
  }


def test_extract_pairs():
  word = (('a', ), ('b', ), ('c', ), ('d', ), (None, ))
  pairs = util.extract_pairs(word)
  assert pairs == [
      (('a', ), ('b', )),
      (('b', ), ('c', )),
      (('c', ), ('d', )),
      (('d', ), (None, )),
  ]


def test_binary_pair_encoding_simple():
  vocab = util.build_vocabulary('a b c')
  bpe = util.binary_pair_encoding(vocab, 1)
  assert bpe.token_counts == {
      ('a', None): 1,
      ('b', None): 1,
      ('c', None): 1,
  }


def test_binary_pair_encoding_example():
  vocab = util.build_vocabulary(('low low low low low '
                                 'lower lower '
                                 'newest newest newest newest newest newest '
                                 'widest widest widest'))

  assert vocab == {
      (('l', ), ('o', ), ('w', ), (None, )): 5,
      (('l', ), ('o', ), ('w', ), ('e', ), ('r', ), (None, )): 2,
      (('n', ), ('e', ), ('w', ), ('e', ), ('s', ), ('t', ), (None, )): 6,
      (('w', ), ('i', ), ('d', ), ('e', ), ('s', ), ('t', ), (None, )): 3,
  }

  num_merges = 10
  bpe = util.binary_pair_encoding(vocab, num_merges)

  print()
  pprint(bpe.token_counts)
  print()
  pprint(list(zip(range(len(bpe.all_tokens)), bpe.all_tokens)))
