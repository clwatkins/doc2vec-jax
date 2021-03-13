FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04

SHELL ["/bin/bash", "-c"]
ENV HOME="/root"
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get install -y --no-install-recommends make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
RUN apt-get install -y git

RUN git clone --depth=1 https://github.com/pyenv/pyenv.git $HOME/.pyenv

ENV PYENV_ROOT="$HOME/.pyenv"
ENV PATH="$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH"

RUN echo 'export PYENV_ROOT="~/.pyenv"' >> $HOME/.bashrc
RUN echo 'export PATH="$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH"' >> $HOME/.bashrc
RUN source $HOME/.bashrc

RUN pyenv install 3.8.7
RUN pyenv global 3.8.7

RUN pip install poetry

COPY poetry.lock pyproject.toml doc2vec-jax/
WORKDIR doc2vec-jax/

RUN poetry install

COPY ./doc2vec ./doc2vec/