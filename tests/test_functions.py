from chainer_transformer.functions import generate_positional_encodings

import pytest


def test_generate_positional_encoding():
    start = 0
    end = 100
    dim = 256
    l = end - start

    output = generate_positional_encodings(start, end, dim)

    assert output.shape == (l, dim)
