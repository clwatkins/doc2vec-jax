"""Implement key Doc2Vec model variants."""

import haiku as hk
import jax
import jax.numpy as jnp


class PVDM(hk.Module):
    """Implements a PV-DM Doc2Vec model."""

    def __init__(self, word_vocab_size: int, doc_vocab_size: int,
                 embedding_size: int, window_size: int, context_mode: str, name: str):
        super().__init__(name=name)
        self.word_embedder = hk.Embed(
            vocab_size=word_vocab_size,
            embed_dim=embedding_size, name='word_embeddings')
        self.doc_embedder = hk.Embed(
            vocab_size=doc_vocab_size,
            embed_dim=embedding_size, name='doc_embeddings')
        self.fc = hk.Linear(
            output_size=word_vocab_size, name='fully_connected')

        self.window_size = window_size
        self.embedding_size = embedding_size
        self.context_mode = context_mode

    def __call__(self, doc_id, context_words):
        doc_embedding = self.doc_embedder(doc_id)
        word_embeddings = self.word_embedder(context_words)

        doc_embedding_expanded = jnp.expand_dims(doc_embedding, axis=1)

        doc_and_word_embeddings = jnp.concatenate(
            [doc_embedding_expanded, word_embeddings], axis=1)

        if self.context_mode == 'concat':
            flattened = jnp.reshape(doc_and_word_embeddings, (-1, (self.window_size + 1) * self.embedding_size))
        elif self.context_mode == 'average':
            flattened = jnp.mean(doc_and_word_embeddings, axis=1)  # Average across dim concatting doc + word vectors
        else:
            raise ValueError('context_mode must be set to either `concat` or `average`')

        logits = self.fc(flattened)

        return jax.nn.softmax(logits)
