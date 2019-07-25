#!/usr/bin/env bash

cd docker/dataset/
NO_SUDO=$(groups | grep -E "(docker )|( docker)|(^docker$)")
if [[ $NO_SUDO ]]; then
  docker build -t moses-tokenizer .
else
  sudo docker build -t moses-tokenizer .
fi
