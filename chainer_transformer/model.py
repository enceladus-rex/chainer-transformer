from math import sqrt
from collections import namedtuple

from chainer import Link, Chain, ChainList

import chainer.functions as F
import chainer.links as L

try:
  import cupy as xp
except ImportError:
  import numpy as xp

HeadData = namedtuple('HeadData', ('query', 'key', 'value'))


def scaled_dot_product_attention(queries, keys, values, scale=0.1, mask=None):
  x1 = F.matmul(queries, keys, transb=True) / xp.array(scale, dtype=keys.dtype)
  x2 = x1 * mask if mask is not None else x1
  x3 = F.softmax(x2)
  x4 = F.matmul(x3, values)
  return x4


def generate_positional_encoding(start, end, dim):
  positions = xp.arange(start, end)
  stacks = []
  for i in range(dim):
    divisor = 10000**(2 * i / float(dim))
    elements = positions / divisor
    if i % 2 == 0:
      pe = xp.sin(elements)
    else:
      pe = xp.cos(elements)
    stacks.append(pe)
  return xp.transpose(xp.stack(stacks))


class MultiHeadAttention(Chain):
  def __init__(self, num_heads, model_dim, key_dim, value_dim):
    super().__init__()
    self.num_heads = num_heads
    self.model_dim = model_dim
    self.key_dim = key_dim
    self.value_dim = value_dim
    self.multi_head_dim = num_heads * value_dim
    with self.init_scope():
      self.head_query_links = ChainList()
      self.head_key_links = ChainList()
      self.head_value_links = ChainList()
      for i in range(num_heads):
        self.head_query_links.append(L.Linear(model_dim, key_dim))
        self.head_key_links.append(L.Linear(model_dim, key_dim))
        self.head_value_links.append(L.Linear(model_dim, value_dim))
      self.output_link = L.Linear(self.multi_head_dim, model_dim)

  def forward(self, queries, keys, values, mask=None):
    heads = []
    for i in range(self.num_heads):
      query_projection = self.head_query_links[i](queries)
      key_projection = self.head_key_links[i](keys)
      value_projection = self.head_value_links[i](values)

      head = scaled_dot_product_attention(
        query_projection, key_projection, value_projection, mask=mask)

      heads.append(head)

    multi_head = F.concat(heads)
    return self.output_link(multi_head)


class PointwiseFeedForwardNetwork(Chain):
  def __init__(self, model_dim, inner_dim):
    super().__init__()
    with self.init_scope():
      self.lin1 = L.Linear(model_dim, inner_dim)
      self.lin2 = L.Linear(inner_dim, model_dim)

  def forward(self, x):
    return self.lin2(F.relu(self.lin1(x)))


class TransformerEncoderUnit(Chain):
  def __init__(self, num_heads, model_dim, ff_dim, p_drop):
    super().__init__()
    self.p_drop = p_drop
    with self.init_scope():
      kv_dim = model_dim // num_heads
      self.mha = MultiHeadAttention(num_heads, model_dim, kv_dim, kv_dim)
      self.lnorm1 = L.LayerNormalization()
      self.lnorm2 = L.LayerNormalization()
      self.ff = PointwiseFeedForwardNetwork(model_dim, ff_dim)

  def forward(self, inputs_unit):
    x1 = F.dropout(self.mha(inputs_unit, inputs_unit, inputs_unit),
                   self.p_drop)
    x2 = self.lnorm1(inputs_unit + x1)
    x3 = F.dropout(self.ff(x2), self.p_drop)
    x4 = self.lnorm1(x2 + x3)
    return x4


class TransformerDecoderUnit(Chain):
  def __init__(self, num_heads, model_dim, ff_dim, p_drop):
    super().__init__()
    self.p_drop = p_drop
    with self.init_scope():
      kv_dim = model_dim // num_heads
      self.mmha = MultiHeadAttention(num_heads, model_dim, kv_dim, kv_dim)
      self.mha = MultiHeadAttention(num_heads, model_dim, kv_dim, kv_dim)
      self.lnorm1 = L.LayerNormalization()
      self.lnorm2 = L.LayerNormalization()
      self.lnorm3 = L.LayerNormalization()
      self.ff = PointwiseFeedForwardNetwork(model_dim, ff_dim)

  def forward(self, inputs_encoding, outputs_unit):
    mask_shape = list(inputs_encoding.shape)
    mask_shape[-1] = mask_shape[-2]
    mask = xp.tril(xp.ones(mask_shape)).astype(inputs_encoding.dtype)
    x1 = F.dropout(self.mmha(outputs_unit, outputs_unit, outputs_unit, mask),
                   self.p_drop)
    x2 = self.lnorm1(outputs_unit + x1)
    x3 = F.dropout(self.mha(inputs_encoding, inputs_encoding, x2), self.p_drop)
    x4 = self.lnorm2(x2 + x3)
    x5 = F.dropout(self.ff(x4), self.p_drop)
    x6 = self.lnorm3(x4 + x5)
    return x6


class TransformerEncoder(Chain):
  def __init__(self, depth, num_heads, model_dim, ff_dim, p_drop):
    super().__init__()
    with self.init_scope():
      self.unit_links = ChainList()
      for i in range(depth):
        self.unit_links.append(
            TransformerEncoderUnit(num_heads, model_dim, ff_dim, p_drop))

  def forward(self, inputs_encoding):
    unit_inputs = [inputs_encoding]
    for unit_link in self.unit_links:
      x = unit_inputs[-1]
      o = unit_link(x)
      unit_inputs.append(o)
    return unit_inputs[-1]


class TransformerDecoder(Chain):
  def __init__(self, depth, num_heads, model_dim, ff_dim, p_drop):
    super().__init__()
    with self.init_scope():
      self.lin1 = L.Linear(model_dim)
      self.unit_links = ChainList()
      for i in range(depth):
        self.unit_links.append(
            TransformerDecoderUnit(num_heads, model_dim, ff_dim, p_drop))

  def forward(self, inputs_encoding, outputs_unit):
    unit_inputs = [outputs_unit]
    for unit_link in self.unit_links:
      x = unit_inputs[-1]
      o = unit_link(inputs_encoding, x)
      unit_inputs.append(o)
    unit_output = unit_inputs[-1]
    return F.softmax(self.lin1(unit_output))


class Transformer(Chain):
  def __init__(self,
               model_dim=512,
               num_heads=8,
               encoder_depth=6,
               decoder_depth=6,
               ff_dim=2048,
               p_drop=0.1):
    super().__init__()
    self.encoder_depth = encoder_depth
    self.decoder_depth = decoder_depth
    self.num_heads = num_heads
    self.model_dim = model_dim
    self.ff_dim = ff_dim
    self.p_drop = p_drop
    self.multiplier = sqrt(model_dim)
    with self.init_scope():
      self.encoder = TransformerEncoder(encoder_depth, num_heads, model_dim,
                                        ff_dim, p_drop)
      self.decoder = TransformerDecoder(decoder_depth, num_heads, model_dim,
                                        ff_dim, p_drop)
      self.linear_embedding = L.Linear(None, model_dim, nobias=True)

  def forward(self, inputs, outputs, inputs_position, outputs_position):
    assert inputs.shape == outputs.shape

    inputs_positional_encoding = generate_positional_encoding(
        inputs_position, inputs_position + inputs.shape[-2], inputs.shape[-1])

    outputs_positional_encoding = generate_positional_encoding(
        outputs_position, outputs_position + outputs.shape[-2],
        outputs.shape[-1])

    input_embedding = self.linear_embedding(inputs) * self.multiplier
    output_embedding = self.linear_embedding(outputs) * self.multiplier

    transformed_inputs = F.dropout(
        inputs_positional_encoding + input_embedding, self.p_drop)
    transformed_ouputs = F.dropout(
        outputs_positional_encoding + output_embedding, self.p_drop)

    encoding = self.encoder(transformed_inputs)
    decoding = self.decoder(encoding, transformed_ouputs)

    inverse_embedding = F.linear(decoding,
                                 F.transpose(self.linear_embedding.W))
    next_token_probabilities = F.softmax(inverse_embedding)

    return next_token_probabilities