FROM nvidia/cuda:11.2.1-cudnn8-runtime-ubuntu20.04
FROM python:3.8

RUN pip install poetry

COPY ./doc2vec doc2vec-jax/doc2vec/
COPY ./doc2vec poetry.lock pyproject.toml doc2vec-jax/

WORKDIR doc2vec-jax/
RUN poetry install