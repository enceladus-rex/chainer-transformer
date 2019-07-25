#!/usr/bin/env bash

set -e
trap 'echo "Error encountered!"' ERR

mkdir -p dataset
cd dataset
wget http://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz
test $(sha256sum ./training-parallel-europarl-v7.tgz | awk '{print $1}') = 0224c7c710c8a063dfd893b0cc0830202d61f4c75c17eb8e31836103d27d96e7
tar -xvf training-parallel-europarl-v7.tgz
