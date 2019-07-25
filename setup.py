from distutils.core import setup
from os.path import dirname, join

import setuptools


setup(name='chainer-transformer',
      version='0.0.1',
      description='Chainer Neural Network Transformer',
      packages=['chainer_transformer'],
      scripts=[
            'scripts/build_docker_image.sh',
            'scripts/download_dataset.sh',
            'scripts/encode_text.py',
            'scripts/index_tokens.sh',
            'scripts/tokenize_en_de.sh',
            'scripts/train_model.py',
      ])
