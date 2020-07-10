import os
import json
import logging
import click

from chainer.iterators import SerialIterator
from chainer.training.updaters import StandardUpdater
from chainer.training import extensions, Trainer
from chainer.optimizers import Adam
from chainer.links import Classifier

from chainer_transformer import util
from chainer_transformer.dataset import make_dataset

from chainer_transformer.model import Transformer


def train(dataset, batch_size=1, max_epoch=1, out='out', gpu_id=0):
    train_iter = SerialIterator(dataset, batch_size)

    # Decrease model size for single GPU.
    model = Classifier(
        Transformer(
            model_dim=256,
            ff_dim=1024,
            encoder_depth=3,
            decoder_depth=3,
        ))
    if gpu_id >= 0:
        model.to_gpu(gpu_id)

    optimizer = Adam()
    optimizer.setup(model)

    updater = StandardUpdater(train_iter, optimizer, device=gpu_id)
    trainer = Trainer(updater, (max_epoch, 'epoch'), out=out)
    trainer.extend(extensions.LogReport())
    trainer.extend(
        extensions.snapshot(filename='snapshot_epoch-{.updater.epoch}'))
    trainer.extend(
        extensions.snapshot_object(model.predictor,
                                   filename='model_epoch-{.updater.epoch}'))
    trainer.extend(
        extensions.PrintReport([
            'epoch', 'main/loss', 'main/accuracy', 'validation/main/loss',
            'validation/main/accuracy', 'elapsed_time'
        ]))
    trainer.extend(
        extensions.PlotReport(['main/loss', 'validation/main/loss'],
                              x_key='epoch',
                              file_name='loss.png'))
    trainer.extend(
        extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'],
                              x_key='epoch',
                              file_name='accuracy.png'))
    trainer.extend(extensions.DumpGraph('main/loss'))

    trainer.run()


@click.command()
@click.option('--source_bpe',
              '-s',
              required=True,
              help='The source dataset BPE filename')
@click.option('--target_bpe',
              '-t',
              required=True,
              help='The target dataset BPE filename')
@click.option('--source_vocab',
              '-s',
              required=True,
              help='The source BPE vocab filename')
@click.option('--target_vocab',
              '-t',
              required=True,
              help='The target BPE vocab filename')
@click.option('--chunk_length',
              '-c',
              default=1000,
              help='The chunk length of a sentence during training')
@click.option('--batch_size', '-b', default=1)
@click.option('--max_epoch', '-m', default=1)
@click.option('--gpu_id', '-g', type=int, default=None)
@click.option('--out', '-o', default='out')
@click.option('--log_level',
              '-l',
              default='INFO',
              type=click.Choice(['DEBUG', 'INFO', 'WARN', 'ERROR']))
def train_model(source_bpe, target_bpe, source_vocab, target_vocab,
                chunk_length, batch_size, max_epoch, gpu_id, out, log_level):
    ll = getattr(logging, log_level)
    logging.getLogger().setLevel(ll)
    if not os.path.exists(out):
        os.makedirs(out)

    gpu_id = gpu_id if gpu_id is not None else -1

    dataset = make_dataset(source_bpe, target_bpe, source_vocab, target_vocab,
                           chunk_length)
    train(dataset, batch_size, max_epoch, out, gpu_id)


if __name__ == '__main__':
    train_model()
