from chainer_transformer import model
import pytest

try:
  import cupy as xp
  use_gpu = True
except ImportError:
  import numpy as xp
  use_gpu = False


def test_multi_head_attention():
  num_queries = 32
  num_kv = 64
  num_heads = 4
  model_dim = 128
  key_dim = 32
  value_dim = 32
  
  mha = model.MultiHeadAttention(num_heads, model_dim, key_dim, value_dim)
  if use_gpu: mha.to_gpu()
  
  queries = xp.random.randn(num_queries, model_dim).astype(xp.float32)
  keys = xp.random.randn(num_kv, model_dim).astype(xp.float32)
  values = xp.random.randn(num_kv, model_dim).astype(xp.float32)
  
  output = mha(queries, keys, values)
  assert output.shape == (num_queries, model_dim)


def test_pointwise_ff():
  batch_size = 20
  model_dim = 128
  inner_dim = 512
  
  x = xp.random.randn(batch_size, model_dim).astype(xp.float32)
  pffn = model.PointwiseFeedForwardNetwork(model_dim, inner_dim)
  if use_gpu: pffn.to_gpu()

  output = pffn(x)
  assert output.shape == (batch_size, model_dim)


def test_transformer_encoder_unit():
  num_heads = 4
  model_dim = 128
  ff_dim = 1024
  p_drop = 0.1
  seq_len = 100

  x = xp.random.randn(seq_len, model_dim).astype(xp.float32)
  teu = model.TransformerEncoderUnit(num_heads, model_dim, ff_dim, p_drop)
  if use_gpu: teu.to_gpu()

  output = teu(x)
  assert output.shape == (seq_len, model_dim)


def test_transformer_encoder():
  depth = 3
  num_heads = 4
  model_dim = 128
  ff_dim = 1024
  p_drop = 0.1
  seq_len = 100

  inputs_encoding = xp.random.randn(seq_len, model_dim).astype(xp.float32)
  te = model.TransformerEncoder(depth, num_heads, model_dim, ff_dim, p_drop)
  if use_gpu: te.to_gpu()

  output = te(inputs_encoding)
  assert output.shape == (seq_len, model_dim)


def test_transformer_decoder_unit():
  num_heads = 4
  model_dim = 128
  ff_dim = 1024
  p_drop = 0.1
  seq_len = 100

  inputs_encoding = xp.random.randn(seq_len, model_dim).astype(xp.float32)
  outputs_encoding = xp.random.randn(seq_len, model_dim).astype(xp.float32)
  tdu = model.TransformerDecoderUnit(num_heads, model_dim, ff_dim, p_drop)
  if use_gpu: tdu.to_gpu()

  output = tdu(inputs_encoding, outputs_encoding)
  assert output.shape == (seq_len, model_dim)


def test_transformer_decoder():
  depth = 3
  num_heads = 4
  model_dim = 128
  ff_dim = 1024
  p_drop = 0.1
  seq_len = 100

  inputs_encoding = xp.random.randn(seq_len, model_dim).astype(xp.float32)
  outputs_encoding = xp.random.randn(seq_len, model_dim).astype(xp.float32)
  td = model.TransformerDecoder(depth, num_heads, model_dim, ff_dim, p_drop)
  if use_gpu: td.to_gpu()

  output = td(inputs_encoding, outputs_encoding)
  assert output.shape == (seq_len, model_dim)


def test_transformer():
  model_dim = 512
  num_heads = 8
  encoder_depth = 6
  decoder_depth = 6
  ff_dim = 2048
  p_drop = 0.1
  seq_len = 100

  inputs = xp.random.randn(seq_len, model_dim).astype(xp.float32)
  outputs = xp.random.randn(seq_len, model_dim).astype(xp.float32)
  transformer = model.Transformer(
    model_dim, 
    num_heads, 
    encoder_depth, 
    decoder_depth,
    ff_dim,
    p_drop)
  if use_gpu: transformer.to_gpu()

  output = transformer(inputs, outputs, 0, 10)
  assert output.shape == (seq_len, model_dim)
  

def test_generate_positional_encoding():
  start = 0
  end = 100
  dim = 256
  l = end - start

  output = model.generate_positional_encoding(start, end, dim)

  assert output.shape == (l, dim)