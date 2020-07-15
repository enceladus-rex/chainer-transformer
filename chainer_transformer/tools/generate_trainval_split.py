import os
import json
import logging
import click
import random

from itertools import chain

from math import ceil
from typing import List


def _count_lines(fn):
    with open(fn, 'r') as f:
        count = 0
        for l in f:
            count += 1
        return count


@click.command()
@click.option(
    '--input_filenames',
    nargs=2,
    help='Files that have the same line count from which to generate the splits'
)
@click.option('--train_fraction',
              default=0.8,
              type=float,
              help='The fraction of the original dataset to use for training')
@click.option('--output_dir', help='The location to save the datasets')
def generate_trainval_split(input_filenames: List[str], train_fraction: float,
                            output_dir: str):
    assert len(input_filenames) > 0, 'must provide at least on input filename'
    assert train_fraction >= 0. and train_fraction <= 1., 'train_fraction invalid'

    basenames = [os.path.basename(x) for x in input_filenames]
    assert len(basenames) == len(
        set(basenames)), 'base filenames must be unique'

    output_train_filenames = [
        os.path.join(output_dir, f'{x}.train') for x in basenames
    ]
    output_val_filenames = [
        os.path.join(output_dir, f'{x}.val') for x in basenames
    ]

    for x in chain(output_train_filenames, output_val_filenames):
        assert not os.path.exists(x), f'{x} already exists'

    line_count = _count_lines(input_filenames[0])
    assert line_count == _count_lines(
        input_filenames[1]), 'line counts do not match'

    num_train = ceil(line_count * train_fraction)
    num_val = line_count - num_train

    is_train_flags = [x < num_train for x in range(line_count)]
    random.shuffle(is_train_flags)

    for input_fn, output_train_fn, output_val_fn, in zip(
            input_filenames, output_train_filenames, output_val_filenames):
        with open(input_fn,
                  'r') as input_file, open(output_train_fn,
                                           'w') as output_train_file, open(
                                               output_val_fn,
                                               'w') as output_val_file:
            for i, l in enumerate(input_file):
                assert i < line_count, 'invalid number of lines encountered'
                if i == (line_count - 1) and l and l[-1] == '\n':
                    l = l[:-1]

                if is_train_flags[i]:
                    output_train_file.write(l)
                else:
                    output_val_file.write(l)


if __name__ == '__main__':
    generate_trainval_split()
