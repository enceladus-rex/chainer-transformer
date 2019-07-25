#!/usr/bin/env bash

NO_SUDO=$(groups | grep -E "(docker )|( docker)|(^docker$)")
USER_ID=$(id -u)
RUN=""
if [[ ! $NO_SUDO ]]; then
  RUN="sudo "
fi

${RUN} docker run \
    -it \
    --mount type=bind,src=$(pwd)/dataset/,dst=/dataset \
    moses-tokenizer \
    bash -c "\
    /mosesdecoder/scripts/tokenizer/tokenizer.perl \
    -l en -threads 4 \
    < /dataset/training/europarl-v7.de-en.en \
    > /dataset/training/europarl-v7.de-en.en.tokenized && \
    chown ${USER_ID}:${USER_ID} /dataset/training/europarl-v7.de-en.en.tokenized"

${RUN} docker run \
    -it \
    --mount type=bind,src=$(pwd)/dataset/,dst=/dataset \
    moses-tokenizer \
    bash -c "\
    /mosesdecoder/scripts/tokenizer/tokenizer.perl \
    -l de -threads 4 \
    < /dataset/training/europarl-v7.de-en.de \
    > /dataset/training/europarl-v7.de-en.de.tokenized && \
    chown ${USER_ID}:${USER_ID} /dataset/training/europarl-v7.de-en.de.tokenized"