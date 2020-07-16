#!/usr/bin/env bash

if [ ! $# -eq 2 ]; then
	echo "Usage: $0 text1 text2"
	exit 1
fi

NUM_OPERATIONS=37000

subword-nmt learn-joint-bpe-and-vocab \
	--input $1 $2 -s $NUM_OPERATIONS -o $1.code --write-vocabulary $1.vocab $2.vocab

subword-nmt apply-bpe -c $1.code < $1 > $1.bpe
subword-nmt apply-bpe -c $1.code < $2 > $2.bpe
