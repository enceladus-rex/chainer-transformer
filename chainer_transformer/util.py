from collections import Counter, defaultdict, ChainMap, namedtuple

import logging
import copy
import json

BinaryPairEncoding = namedtuple(
    'BinaryPairEncoding',
    ('all_tokens', 'merged_tokens', 'pair_tokens', 'token_counts'))


def merge_vocab(dst, src):
    for elem in src:
        if elem in dst:
            dst[elem] += src[elem]
        else:
            dst[elem] = src[elem]


def build_vocabulary(s, end=None):
    lines = s.splitlines()
    num_lines = len(lines)
    vocab = {}
    for i, line in enumerate(lines):
        logging.info('parse line {}/{}'.format(i, num_lines))
        line_vocab = dict(
            Counter([
                tuple(tuple(map(tuple, x)) + ((end, ), ))
                for x in line.split()
            ]))
        merge_vocab(vocab, line_vocab)
    logging.info('built vocabulary with {} words'.format(len(vocab)))
    return vocab


def extract_pairs(word):
    p = []
    for i in range(len(word) - 1):
        p.append((word[i], word[i + 1]))
    return p


def binary_pair_encoding(input_vocab, num_merges):
    vocab = copy.deepcopy(input_vocab)
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

    original_pair_counts = copy.deepcopy(pair_counts)

    merge_tokens = []
    for i in range(num_merges):
        if not pair_counts: break
        max_pair = max(pair_counts, key=pair_counts.get)
        pair_token = max_pair[0] + max_pair[1]
        pair_word_indices = set(pairs[max_pair])
        logging.info('merge {} pair {} count {}'.format(
            i, max_pair, len(pair_word_indices)))
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
            new_token_indices = set()
            m = 0
            while m < len(word):
                if m in merge_indices:
                    new_token = word[m] + word[m + 1]
                    assert new_token == pair_token
                    new_token_indices.add(len(new_word_list))
                    new_word_list.append(new_token)
                    pairs[max_pair].remove(j)
                    pair_counts[max_pair] = max(pair_counts[max_pair] - 1, 0)
                    if pair_counts[max_pair] == 0:
                        assert len(pairs[max_pair]) == 0
                        pair_counts.pop(max_pair)

                    # Remove the existing pairs connected to this one.
                    if m > 0 and (m - 2) not in merge_indices:
                        left_pair = (word[m - 1], word[m])
                        pairs[left_pair].remove(j)
                        pair_counts[left_pair] = max(
                            pair_counts[left_pair] - 1, 0)
                        if pair_counts[left_pair] == 0:
                            assert len(pairs[left_pair]) == 0
                            pair_counts.pop(left_pair)

                    if m < len(word) - 2:
                        right_pair = (word[m + 1], word[m + 2])
                        pairs[right_pair].remove(j)
                        pair_counts[right_pair] = max(
                            pair_counts[right_pair] - 1, 0)
                        if pair_counts[right_pair] == 0:
                            assert len(pairs[right_pair]) == 0
                            pair_counts.pop(right_pair)

                    m += 1
                else:
                    new_word_list.append(word[m])
                m += 1

            new_word = tuple(new_word_list)

            # Add new pairs.
            for index in new_token_indices:
                if index > 0 and (index - 1) not in new_token_indices:
                    new_left_pair = (new_word[index - 1], new_word[index])
                    pairs[new_left_pair].append(j)
                    pair_counts[new_left_pair] += 1

                if index < len(new_word) - 1:
                    new_right_pair = (new_word[index], new_word[index + 1])
                    pairs[new_right_pair].append(j)
                    pair_counts[new_right_pair] += 1

            # assert max_pair in pair_counts
            assert new_word not in vocab

            vocab[new_word] = vocab[words[j]]
            vocab.pop(words[j])
            words[j] = new_word

        if max_pair in pair_counts: pair_counts.pop(max_pair)
        vocab[(pair_token, )] = num_pairs
        merge_tokens.append(pair_token)

    cm = ChainMap(vocab, original_pair_counts)
    token_counts = {sum(x, ()): y for x, y in cm.items()}
    tokens = list(token_counts.keys())
    tokens.sort(key=token_counts.get, reverse=True)

    merge_tokens.sort(key=token_counts.get, reverse=True)
    original_pair_tokens = list(
        sum(x, ()) for x in original_pair_counts.keys())
    original_pair_tokens.sort(key=token_counts.get, reverse=True)

    return BinaryPairEncoding(
        all_tokens=tokens,
        merged_tokens=merge_tokens,
        pair_tokens=original_pair_tokens,
        token_counts=token_counts,
    )


def serialize_tokens(t):
    return json.dumps(t, ensure_ascii=False)


def deserialize_tokens(s):
    return [tuple(x) for x in json.loads(s)]


class TokenTrie:
    def __init__(self, value=None, has_value=True):
        self.value = value
        self.has_value = has_value
        self.children = {}

    def insert(self, key, value):
        assert len(key) > 0
        current_node = self
        for e in key:
            if e not in current_node.children:
                node = TokenTrie(has_value=False)
                current_node.children[e] = node
            current_node = current_node.children[e]
        if current_node.has_value: return None
        current_node.has_value = True
        current_node.value = value
        return current_node

    def lookup(self, key):
        assert len(key) > 0
        current_node = self
        for e in key:
            if e not in current_node.children:
                return None, False
            current_node = current_node.children[e]
        if not current_node.has_value:
            return None, False
        return current_node.value, True


def build_token_trie(tokens):
    root = TokenTrie(has_value=False)
    for i, token in enumerate(tokens):
        root.insert(token, i)
    return root


def make_words(s):
    v = s.split()
    words = [tuple(x) + (None, ) for x in v]
    return words


def word_token_indices(word, trie):
    assert len(word) > 0
    valid_tokens = []
    current_node = trie
    for i, elem in enumerate(word):
        if elem in current_node.children:
            next_node = current_node.children[elem]
            if next_node.has_value:
                valid_tokens.append((i, next_node))
            current_node = next_node
        else:
            break

    reverse_tokens = reversed(valid_tokens)
    for index, node in reverse_tokens:
        remaining_elems = word[index + 1:]
        if not remaining_elems:
            assert node.has_value
            return [node.value]

        next_token_indices = word_token_indices(remaining_elems, trie)
        if next_token_indices is not None:
            return [node.value] + next_token_indices

    return None
