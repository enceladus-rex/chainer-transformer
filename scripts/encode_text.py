#!/usr/bin/env python

import os
import json
import logging
import click

from chainer_transformer import util


@click.command()
@click.option('--text', '-t', multiple=True)
@click.option('--num_merges', '-n', default=37000)
@click.option('--output', '-o')
@click.option('--log_level', '-l', default='INFO', 
  type=click.Choice(['DEBUG', 'INFO', 'WARN', 'ERROR']))
def encode(text, num_merges, output, log_level):
  ll = getattr(logging, log_level)
  logging.getLogger().setLevel(ll)
  if not os.path.exists(os.path.dirname(output)):
    logging.error('Output directory is invalid')
    return 1

  assert num_merges >= 0

  if len(text) == 0:
    logging.error('Must provide at least one text file')
    return 1

  text_objects = []
  for t in text:
    with open(t, 'r') as f:
      text_objects.append(f.read())
  
  full_text = '\n'.join(text_objects)
  vocab = util.build_vocabulary(full_text)
  bpe = util.binary_pair_encoding(vocab, num_merges)

  tokens = bpe.pair_tokens + bpe.merged_tokens
  with open(output, 'w+') as f:
    f.write(util.serialize_tokens(tokens))


if __name__ == '__main__':
  encode()