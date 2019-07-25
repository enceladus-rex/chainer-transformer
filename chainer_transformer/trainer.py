from chainer.iterators import SerialIterator
from chainer.training.updaters import StandardUpdater
from chainer.training import extensions, Trainer
from chainer.optimizers import Adam
from chainer.links import Classifier

from chainer_transformer.model import Transformer


def train(dataset, batch_size=1, max_epoch=1, out='result', gpu_id=0):
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
    model.to_gpu()

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
