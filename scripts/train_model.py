#!/usr/bin/env python

import os
import json
import logging
import click

from chainer_transformer import util
from chainer_transformer.dataset import make_dataset
from chainer_transformer.trainer import train


@click.command()
@click.option('--source', '-s', required=True)
@click.option('--target', '-t', required=True)
@click.option('--tokens', '-k', required=True)
@click.option('--chunk_length', '-c', default=100)
@click.option('--batch-size', '-b', default=1)
@click.option('--max_epoch', '-m', default=1)
@click.option('--use_gpu/--no_gpu', '-g/-n', default=True)
@click.option('--out', '-o', default='result')
@click.option('--log_level', '-l', default='INFO', 
  type=click.Choice(['DEBUG', 'INFO', 'WARN', 'ERROR']))
def train_model(
    source, target, tokens, chunk_length,
    batch_size, max_epoch, use_gpu, out, log_level):
  ll = getattr(logging, log_level)
  logging.getLogger().setLevel(ll)
  if not os.path.exists(out):
    os.makedirs(out)

  with open(tokens, 'r') as f:
    tokens = util.deserialize_tokens(f.read())
  
  token_trie = util.build_token_trie(tokens)
  dim = len(tokens) + 1
  eol_index = len(tokens)

  gpu_id = 0 if use_gpu else -1
  
  dataset = make_dataset(
    source, target, token_trie, dim, eol_index, chunk_length)
  train(dataset, batch_size, max_epoch, out, gpu_id)


if __name__ == '__main__':
  train_model()