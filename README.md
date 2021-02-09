# Doc2Vec JAX

This is a JAX-based implementation of Le and Mikolov's [Doc2Vec algorithm](https://arxiv.org/abs/1405.4053), which 
builds on Word2Vec to generate document-level (bag of words) representations.

The paper proposes 2 main model variants -- Paragraph Vector-Distributed Memory (`PV-DM`) and Distributed Bag of Words (`DBOW`) -- although found `PV-DM` was more performant in most situations.

Currently, we only implement `PV-DM`, which extends Word2Vec's `CBOW` method to co-train document and word embeddings:

![Doc2Vec PV-DM](resources/pvdm_diagram.png)

## Codebase

- `doc2vec` contains the core implementation, including code to prepare documents for training
- `experiments` contains key experiments described in the original paper, reimplemented here

For reference, we reach the performance mentioned.

## TODO

- [] `DBOW` model variant
- [] Negative sampling