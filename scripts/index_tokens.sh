#!/usr/bin/env bash

./scripts/encode_text.py \
  -t ./dataset/training/europarl-v7.de-en.en.tokenized \
  -t ./dataset/training/europarl-v7.de-en.de.tokenized \
  -n 37000 \
  -o ./dataset/training/tokens.de-en.json