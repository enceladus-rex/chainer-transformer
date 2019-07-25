#!/usr/bin/env bash

cd docker/dataset/
NO_SUDO=$(groups | grep -E "(docker )|( docker)|(^docker$)")
RUN=""
if [[ ! $NO_SUDO ]]; then
  RUN="sudo "
fi

$RUN docker build -t moses-tokenizer .