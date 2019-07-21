from collections import Counter, defaultdict


def build_vocabulary(s, end='</w>'):
  return dict(Counter(s.split().map(
    lambda x: tuple(list(x) + [end]))
  )))


def extract_pairs(word, end=None):
  p = []
  for i in range(len(word)-1):
    p.append((word[i], word[i-1]))

  if word:
    p.append((word[-1], end))


def binary_pair_encoding(vocab):
  pairs = defaultdict(int)
  for word in vocab:
    word_pairs = extract_pairs(word)