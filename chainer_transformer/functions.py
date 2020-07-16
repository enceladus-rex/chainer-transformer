from typing import NamedTuple, List, Union, Tuple, Any

import chainer

import chainer.functions as F
import chainer.links as L

try:
    import cupy as xp
except ImportError:
    import numpy as xp


def scaled_dot_product_attention(queries, keys, values, scale=1., mask=None):
    x1 = F.matmul(queries, keys, transb=True) * xp.array(scale,
                                                         dtype=keys.dtype)
    x2 = F.where(mask,
                 xp.ones_like(x1.array) *
                 -xp.inf, x1) if mask is not None else x1
    x3 = F.softmax(x2)
    x4 = F.matmul(x3, values)
    return x4


def generate_positional_encodings(start, end, dim):
    positions = xp.arange(start, end)
    stacks = []
    for i in range(dim):
        divisor = 10000**(2 * i / float(dim))
        elements = positions / divisor
        if i % 2 == 0:
            pe = xp.sin(elements, dtype=xp.float32)
        else:
            pe = xp.cos(elements, dtype=xp.float32)
        stacks.append(pe)
    return xp.transpose(xp.stack(stacks))


class _NestedStructure(NamedTuple):
    tuple_constructor: type
    data: List[Union['_NestedStructure', List[Any]]]


def _nested_tuple_constructor(x):
    t = type(x)
    if t is tuple:
        t = lambda *y: tuple(y)
    return t


def _insert_nested_structure(data: List, item: Tuple):
    for x in item:
        if isinstance(x, tuple):
            t = _nested_tuple_constructor(x)
            n = _NestedStructure(t, [])
            _insert_nested_structure(n.data, x)
            data.append(n)
        else:
            data.append([])


def _build_nested_structures(x):
    if isinstance(x, tuple):
        n = _NestedStructure(_nested_tuple_constructor(x), [])
        _insert_nested_structure(n.data, x)
        return n
    else:
        return []


def _add_nested_item(n, item):
    if isinstance(n, _NestedStructure):
        for d, x in zip(n.data, item):
            _add_nested_item(d, x)
    else:
        n.append(item)


def _stack_nested_structure(structure):
    if isinstance(structure, _NestedStructure):
        args = [_stack_nested_structure(x) for x in structure.data]
        return structure.tuple_constructor(*args)
    else:
        return chainer.as_variable(xp.stack([xp.asarray(x) for x in structure]))


def stack_nested(data):
    if not data:
        return ()

    structure = _build_nested_structures(data[0])

    for x in data:
        _add_nested_item(structure, x)

    return _stack_nested_structure(structure)

