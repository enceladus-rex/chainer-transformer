FROM ubuntu:16.04

# Install:
# 1. Moses dependencies.
RUN apt-get update -y && \
    apt-get install -y \
      g++ \
      git \
      subversion \
      automake \
      libtool \
      zlib1g-dev \
      libboost-all-dev \
      libbz2-dev \
      liblzma-dev \
      python-dev \
      graphviz \
      imagemagick \
      make \
      cmake \
      libgoogle-perftools-dev \
      autoconf \
      doxygen && \
    apt-get upgrade -y

RUN git clone https://github.com/moses-smt/mosesdecoder /mosesdecoder