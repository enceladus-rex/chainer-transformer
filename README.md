# Chainer Transformer (Neural Sequence Transduction)

This a paper implementation of `Attention is All You Need by Vaswani et al.` using the Chainer Neural Network library for
learning purposes. 

## Instructions

To avoid installing the many dependencies needed by the [mosesdecoder](https://github.com/moses-smt/mosesdecoder), you
can use `docker` instead. Just make sure to have [installed it first](https://docs.docker.com/install/).

You can setup the environment by downloading the dataset and generating the training tokens. You can run the provided scripts
from the root of the project to do this:

```
./scripts/download_dataset.sh
./scripts/build_docker_image.sh
./scripts/tokenize_en_de.sh
./scripts/index_tokens.sh
```

Now you can use the `trainer` to train the model:

```
./scripts/train_model.py --help
```

Note that the `source` and `target` files should be the moses tokenized ones.
