from collections import Counter, defaultdict


def build_vocabulary(s, end=None):
  return dict(Counter([tuple(tuple(map(tuple, x)) + ((end,),)) for x in s.split()]))


def extract_pairs(word):
  p = []
  for i in range(len(word)-1):
    p.append((word[i], word[i+1]))
  return p


def binary_pair_encoding(vocab, num_merges):
  words = list(vocab.keys())
  word_index_mapping = {}
  for i, word in enumerate(words):
    word_index_mapping[word] = i

  pairs = defaultdict(list)
  for word in vocab:
    word_pairs = extract_pairs(word)
    for p in word_pairs:
      pairs[p].append(word_index_mapping[word])

  pair_counts = defaultdict(int)
  for p in pairs:
    pair_counts[p] = len(pairs[p])

  for i in range(num_merges):
    max_pair = max(pair_counts, key=pair_counts.get)
    pair_token = max_pair[0] + max_pair[1]
    pair_word_indices = set(pairs[max_pair])
    num_pairs = 0
    for j in pair_word_indices:
      word = words[j]
      word_pairs = extract_pairs(word)
      k = 0
      merge_indices = set()
      while k < len(word_pairs):
        if word_pairs[k] == max_pair:
          merge_indices.add(k)
          num_pairs += 1
          k += 1
        k += 1

      new_word_list = []
      m = 0
      while m < len(word):
        if m in merge_indices:
          new_token = word[m] + word[m+1]
          assert new_token == pair_token
          new_word_list.append(new_token)

          # Remove the existing pairs connected to this one and add new ones.
          if m > 0:
            left_pair = (word[m-1], word[m])
            pairs[left_pair].remove(j)
            pair_counts[left_pair] = max(pair_counts[left_pair]-1, 0)

            new_left_pair = (word[m-1], new_token)
            pairs[new_left_pair].append(j)
            pair_counts[new_left_pair] += 1
          
          if m < len(word) - 1:
            right_pair = (word[m], word[m+1])
            pairs[right_pair].remove(j)
            pair_counts[right_pair] = max(pair_counts[right_pair]-1, 0)

            new_right_pair = (word[m-1], new_token)
            pairs[new_right_pair].append(j)
            pair_counts[new_right_pair] += 1

          m += 1
        else:
          new_word_list.append(word[m])
        m += 1
      new_word = tuple(new_word_list)
      words[j] = new_word
    print(max_pair, words)
    vocab.pop(max_pair[0])
    vocab.pop(max_pair[1])
    vocab[pair_token] = num_pairs
  token_indices = vocab.keys()
  token_indices.sort(key=vocab.get)
  return token_indices, vocab