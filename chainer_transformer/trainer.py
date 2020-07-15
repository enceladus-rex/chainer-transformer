import os
import shutil
import json
import logging
import click
import gc
import time

from math import sqrt

import chainer

from chainer.iterators import SerialIterator
from chainer.training.updaters import StandardUpdater
from chainer.training import extensions, Trainer
from chainer.optimizers import Adam
from chainer.links import Classifier
from chainer.serializers import load_npz, save_npz

import chainer.functions as F

from chainer_transformer.dataset import make_dataset, make_vocab
from chainer_transformer.model import Transformer


def stack_batch(data):
    source_embeddings = [x[0][0] for x in data]
    source_masks = [x[0][1] for x in data]
    target_embeddings = [x[1][0] for x in data]
    target_masks = [x[1][1] for x in data]
    return (F.stack(source_embeddings),
            F.stack(source_masks)), (F.stack(target_embeddings),
                                     F.stack(target_masks))


class TrainingState:
    def __init__(self):
        self.step = 0

    def serialize(self, s):
        if isinstance(s, chainer.Serializer):
            s('step', self.step)
        else:
            self.step = s('step', self.step)


def save_training(out, model, optimizer, state):
    print('saving training')
    model_filename = os.path.join(out, 'transformer.model')
    optimizer_filename = os.path.join(out, 'transformer.optimizer')
    state_filename = os.path.join(out, 'transformer.state')

    for fn in (model_filename, optimizer_filename, state_filename):
        try:
            shutil.copy(fn, fn + '.backup')
        except Exception:
            pass

    save_npz(model_filename, model)
    save_npz(optimizer_filename, optimizer)
    save_npz(state_filename, state)


def load_training(out, model, optimizer, state):
    print('loading training')
    model_filename = os.path.join(out, 'transformer.model')
    optimizer_filename = os.path.join(out, 'transformer.optimizer')
    state_filename = os.path.join(out, 'transformer.state')

    for obj, fn in ((model, model_filename), (optimizer, optimizer_filename),
                    (state, state_filename)):
        if os.path.exists(fn):
            load_npz(fn, obj)


@click.command()
@click.option('--source_bpe',
              '-s',
              required=True,
              help='The source dataset BPE filename')
@click.option('--target_bpe',
              '-t',
              required=True,
              help='The target dataset BPE filename')
@click.option('--source_glove',
              '-v',
              required=True,
              help='The source BPE glove numpy filename')
@click.option('--target_glove',
              '-w',
              required=True,
              help='The target BPE glove numpy filename')
@click.option('--chunk_length',
              '-c',
              default=64,
              help='The chunk length of a sentence during training')
@click.option('--batch_size', '-b', default=8)
@click.option('--warmup_steps', default=4000)
@click.option('--save_decimation', default=100)
@click.option('--num_steps', '-m', default=1000)
@click.option('--gpu_id', '-g', type=int, default=None)
@click.option('--out', '-o', default='result')
@click.option('--log_level',
              '-l',
              default='INFO',
              type=click.Choice(['DEBUG', 'INFO', 'WARN', 'ERROR']))
def train_model(source_bpe, target_bpe, source_glove, target_glove,
                chunk_length, batch_size, warmup_steps, save_decimation,
                num_steps, gpu_id, out, log_level):
    ll = getattr(logging, log_level)
    logging.getLogger().setLevel(ll)
    if not os.path.exists(out):
        os.makedirs(out)

    gpu_id = gpu_id if gpu_id is not None else -1

    device_name = '@intel64'
    if gpu_id >= 0:
        device_name = f'@cupy:{gpu_id}'

    with chainer.using_device(device_name):
        source_vocab = make_vocab(source_glove)
        target_vocab = make_vocab(target_glove)
        output_model_dim = target_vocab.embedding_size
        dataset = make_dataset(source_bpe, target_bpe, source_vocab,
                               target_vocab, chunk_length)
        train_iter = SerialIterator(dataset, batch_size)
        state = TrainingState()
        model = Transformer(source_vocab, target_vocab)
        model.to_gpu(gpu_id)
        optimizer = Adam(beta1=0.99, beta2=0.98, eps=1e-9).setup(model)

        load_training(out, model, optimizer, state)

        try:
            for n, batch in enumerate(train_iter):
                if n >= num_steps:
                    break

                if (n + 1) % save_decimation == 0:
                    save_training(out, model, optimizer, state)

                model.cleargrads()
                gc.collect()
                source, target = stack_batch(batch)
                output_probs, output_masks = model(source[0],
                                                   source[1],
                                                   length=chunk_length)
                loss = F.softmax_cross_entropy(
                    F.reshape(output_probs,
                              (output_probs.shape[0] * output_probs.shape[1],
                               output_probs.shape[2])),
                    F.reshape(target[0],
                              (target[0].shape[0] * target[0].shape[1], )))
                loss.backward()

                learning_rate = sqrt(output_model_dim) * min(
                    sqrt(state.step), state.step * (warmup_steps**-1.5))
                optimizer.alpha = learning_rate
                optimizer.update()

                print('time =', int(time.time()), '| step =', state.step,
                      '| loss =', loss, '| lr =', learning_rate)

                state.step += 1
        finally:
            save_training(out, model, optimizer, state)


if __name__ == '__main__':
    train_model()
