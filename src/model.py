from chainer import Link, Chain
from chainer.links import LayerNormalization
from chainer.functions import softmax


class MultiHeadSelfAttention(Chain):
  
  def __init__(self, query_dim, key_dim, value_dim):
    pass


class TransformerEncoder(Chain):

  def __init__(self, depth):
    pass


class MaskedMultiHeadAttention(Chain):
  
  def __init__(self):
    pass


class TransformerDecoder(Chain):

  def __init__(self):
    pass


class Transformer(Chain):

  def __init__(self):
    pass