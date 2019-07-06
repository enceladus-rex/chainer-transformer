from math import sqrt
from collections import namedtuple

from chainer import Link, Chain

import chainer.functions as F
import chainer.links as L


HeadData = namedtuple('HeadData', ('query', 'key', 'value'))


def scaled_dot_product_attention(queries, keys, values):
  x1 = F.matmul(queries, keys, transb=True) / self.scale
  x2 = F.softmax(x1)
  x3 = F.matmul(x2, values)
  return x3


class MultiHeadAttention(Chain):
  def __init__(self, num_heads, model_dim, key_dim, value_dim):
    super().__init__()
    self.num_heads = num_heads
    self.model_dim = model_dim
    self.key_dim = key_dim
    self.value_dim = value_dim
    self.multi_head_dim = num_heads * value_dim
    self.head_links = []
    with self.init_scope():
      for i in range(num_heads):
        self.head_links.append(
          HeadData(
            L.Linear(model_dim, key_dim),
            L.Linear(model_dim, key_dim),
            L.Linear(model_dim, value_dim)))
      self.output_link = L.Linear(self.multi_head_dim, model_dim)

  def forward(self, queries, keys, values):
    projections = []
    heads = []
    for i in range(self.num_heads):
      hl = self.head_links[i]

      projection = HeadData(hl.query(queries), hl.key(keys), hl.value(values))
      head = scaled_dot_product_attention(*projection)

      projections.append(projection)
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
  def __init__(self, num_heads, model_dim, ff_dim):
    with self.init_scope():
      kv_dim = model_dim // num_heads
      self.mha = MultiHeadAttention(num_heads, model_dim, kv_dim, kv_dim)
      self.lnorm1 = L.LayerNormalization()
      self.lnorm2 = L.LayerNormalization()
      self.ff = PointwiseFeedForwardNetwork(model_dim, ff_dim)
  
  def forward(self, inputs_unit):
    x1 = self.mha(inputs_unit, inputs_unit, inputs_unit)
    x2 = self.lnorm1(inputs_unit + x1)
    x3 = self.ff(x2)
    x4 = self.lnorm1(x2 + x3)
    return x4


class TransformerDecoderUnit(Chain):
  def __init__(self, num_heads, model_dim, ff_dim):
    super().__init__()
    with self.init_scope():
      kv_dim = model_dim // num_heads
      self.mmha = MaskedMultiHeadAttention(
        num_heads, model_dim, kv_dim, kv_dim)
      self.mha = MultiHeadAttention(
        num_heads, model_dim, kv_dim, kv_dim)
      self.lnorm1 = L.LayerNormalization()
      self.lnorm2 = L.LayerNormalization()
      self.lnorm3 = L.LayerNormalization()
      self.ff = PointwiseFeedForwardNetwork(model_dim, ff_dim)
  
  def forward(self, inputs_encoding, outputs_unit):
    x1 = self.mmha(outputs_unit, outputs_unit, outputs_unit)
    x2 = self.lnorm1(outputs_unit + x1)
    x3 = self.mha(inputs_encoding, inputs_encoding, x2)
    x4 = self.lnorm2(x2 + x3)
    x5 = self.ff(x4)  # TODO: Determine if an activation is required with this.
    x6 = self.lnorm3(x4 + x5)
    return x6


class TransformerEncoder(Chain):

  def __init__(self, depth, num_heads, model_dim, ff_dim):
    super().__init__()
    self.unit_links = []
    with self.init_scope():
      for i in range(depth):
        self.unit_links.append(
          TransformerEncoderUnit(num_heads, model_dim, ff_dim))
        
  def forward(self, inputs_encoding):
    unit_inputs = [inputs_encoding]
    for unit_link in self.unit_links:
      x = unit_inputs[-1]
      o = unit_link(x)
      unit_inputs.append(o)
    return unit_inputs[-1]


class MaskedMultiHeadAttention(Chain):
  def __init__(self):
    pass


class TransformerDecoder(Chain):
  def __init__(self, depth, num_heads, model_dim, ff_dim):
    super().__init__()
    self.unit_links = []
    with self.init_scope():
      self.lin1 = L.Linear(model_dim)
      for i in range(depth):
        self.unit_links.append(
          TransformerDecoderUnit(num_heads, model_dim, ff_dim))

  def forward(self, inputs_encoding, outputs_unit):
    unit_inputs = [outputs_unit]
    for unit_link in self.unit_links:
      x = unit_inputs[-1]
      o = unit_link(inputs_encoding, x)
      unit_inputs.append(o)
    unit_output = unit_inputs[-1]
    return F.softmax(self.lin1(unit_output))


class Transformer(Chain):
  def __init__(self, depth=6, num_heads=8, model_dim=512, ff_dim=2048):
    super().__init__()
    self.depth = depth
    self.num_heads = num_heads
    self.model_dim = model_dim
    self.ff_dim = ff_dim
    with self.init_scope():
      self.encoder = TransformerEncoder(depth, num_heads, model_dim, ff_dim)
      self.decoder = TransformerDecoder(depth, num_heads, model_dim, ff_dim)
  
  def forward(self, inputs, shifted_outputs):
    # TODO: Generate input embedding.
    # TODO: Add positional encoding.
    inputs_encoding = self.encoder(inputs)
    return self.decoder(inputs_encoding, shifted_outputs)