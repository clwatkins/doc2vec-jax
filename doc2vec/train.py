"""Train model."""
from absl import app
from absl import flags
from absl import logging

import datetime as dt
from pathlib import Path
import pickle
from typing import Any, List, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import jax.profiler
import optax
import numpy as np
import tensorflow as tf
import wandb

from doc2vec.generate_training_data import DATASET_NAME_PATTERN
from doc2vec.models import PVDM, DBOW

FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 128, help='Batch size')
flags.DEFINE_integer('training_epochs', 10, help='Num epochs to train')
flags.DEFINE_integer('embedding_size', 128, help='Embedding size')
flags.DEFINE_enum('architecture', 'dbow', ['pvdm', 'dbow'],
                  help='The model variant to select.')
flags.DEFINE_enum('context_mode', 'average', ['concat', 'average'],
                  help='How to combine context embeddings in the PV-DM model. '
                       'They can either be concatenated (large and slow), or averaged.')
flags.DEFINE_float('learning_rate', 1e-3, help='Optimizer learning rate')
flags.DEFINE_enum('optimizer', 'adam', ['sgd', 'adam'],
                  help='optimizer to use. momentum only applied if SGD')
flags.DEFINE_float('momentum', 0.9, help='Rate at which to decay LR')
flags.DEFINE_integer('log_every', 100, help='How often to log results')
flags.DEFINE_string('model_dir', None, help='Base directory to store models in')
flags.DEFINE_bool('wandb', True,
                  help='Whether to track experiment using W&B')
flags.DEFINE_string('wandb_project', 'doc2vec',
                    help='W&B project name to track experiment under')
flags.DEFINE_integer('random_seed', 1234, help='Seed used to shuffle input data and initialise model')

flags.register_validator(
    flag_name='model_dir',
    checker=lambda fp: Path(fp).expanduser().exists(),
    message='One of your file/directory paths appears not to exist'
)

LOG_FMT_STRING = 'Epoch {epoch} | Batch {batch} | Accuracy {accuracy:.1%} | Loss {loss:.2f}'
MODEL_NAME_PATTERN = 'doc2vec_{dataset_name}_{architecture}_{window_size}window_{vocab_size}vocab_' \
                     '{embedding_size}embeddingdim_{context_mode}context_model_{batch_size}batch_{optimizer}optim_' \
                     '{learning_rate}lr_{momentum}momentum_{training_epochs}epochs'


def _load_dataset_and_vocabs_from_file() -> Tuple[tf.data.Dataset, List[str], List[str]]:
    data_dir = Path(FLAGS.training_data_dir).expanduser() / 'training_data' / \
        DATASET_NAME_PATTERN.format(
            dataset_name=FLAGS.dataset_name,
            architecture=FLAGS.architecture,
            window_size=FLAGS.window_size,
            vocab_size=FLAGS.vocab_size
        )

    doc_ids = np.load(data_dir / 'doc_ids.npy')
    target_words = np.load(data_dir / 'target_words.npy')
    if FLAGS.architecture == 'pvdm':
        context_words = np.load(data_dir / 'context_words.npy')
        ds = tf.data.Dataset.from_tensor_slices(
            (doc_ids, context_words, target_words)
        )
    else:
        # DBOW: insert empty dimension to represent extraneous context_words
        ds = tf.data.Dataset.from_tensor_slices(
            (doc_ids, None, target_words)
        )

    with open(data_dir / 'word_vocab.txt') as f:
        word_vocab = f.read().split('\n')

    with open(data_dir / 'doc_vocab.txt') as f:
        doc_vocab = f.read().split('\n')

    return ds, word_vocab, doc_vocab


def _save_model_to(model_params: Any, directory: Path):
    directory.mkdir(parents=True, exist_ok=True)

    with open(directory / 'model.p', "wb") as f:
        pickle.dump(model_params, f, protocol=pickle.DEFAULT_PROTOCOL)


def _get_similar_terms(comparison_terms: List[str],
                       vocab: List[str],
                       embedding_layer: hk.Embed,
                       top_k: int = 5):
    """Finds similar terms to a given target in vector space."""

    def _get_most_similar_k(embeddings, target_idx: int, top_k: int):
        target_weights = embeddings[target_idx]

        distances = embeddings @ target_weights.reshape(-1, 1)
        closest_k = jnp.argsort(distances, 0).reshape(-1,)[1: top_k]  # skip top term -- self
        return closest_k

    for comparison_term in comparison_terms:
        comparison_idx = vocab.index(comparison_term)
        most_similar = _get_most_similar_k(
            embedding_layer['embeddings'], comparison_idx, top_k + 1)  # add 1 as we skip first

        logging.info('Looking for most similar words to: %s', comparison_term)

        for term_idx in jax.device_get(most_similar):
            logging.info('Similar term %s', vocab[term_idx])


def main(_):
    """Run model training loop."""

    tf.debugging.set_log_device_placement(True)
    tf.config.set_soft_device_placement(True)

    @jax.jit
    def loss_fn(params: hk.Params, batch: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> float:
        def categorical_cross_entropy(probs, targets):
            return -jnp.sum(targets * jnp.log(probs), axis=1)

        doc, context, target = batch
        probs = model.apply(params, doc, context)
        target_one_hot = jax.nn.one_hot(
            target, num_classes=len(word_vocab), dtype=jnp.uint32)
        loss = jnp.mean(categorical_cross_entropy(probs, target_one_hot))
        return loss

    @jax.jit
    def calc_accuracy(params: hk.Params, batch: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> float:
        doc, context, target = batch
        probs = model.apply(params, doc, context)

        predicted_class = jnp.argmax(probs, axis=-1)

        return jnp.mean(predicted_class == target)

    @jax.jit
    def update(model_params: hk.Params, opt_state, batch):
        grads = jax.grad(loss_fn)(model_params, batch)
        updates, opt_state = optimizer.update(grads, opt_state)
        new_model_params = optax.apply_updates(model_params, updates)

        return new_model_params, opt_state

    def model_fn(doc_id, context_words=None):
        if FLAGS.architecture == 'pvdm':
            d2v = PVDM(
                word_vocab_size=len(word_vocab),
                doc_vocab_size=len(doc_vocab),
                embedding_size=FLAGS.embedding_size,
                window_size=FLAGS.window_size,
                context_mode=FLAGS.context_mode,
                name=FLAGS.architecture_name
            )
        else:
            d2v = DBOW(
                word_vocab_size=len(word_vocab),
                doc_vocab_size=len(doc_vocab),
                embedding_size=FLAGS.embedding_size,
                name=FLAGS.architecture_name
            )

        return d2v(doc_id, context_words)

    model_name = MODEL_NAME_PATTERN.format(
        model=FLAGS.architecture,
        dataset_name=FLAGS.dataset_name,
        window_size=FLAGS.window_size,
        vocab_size=FLAGS.vocab_size,
        embedding_size=FLAGS.embedding_size,
        context_mode=FLAGS.context_mode,
        batch_size=FLAGS.batch_size,
        optimizer=FLAGS.optimizer,
        learning_rate=FLAGS.learning_rate,
        momentum=FLAGS.momentum,
        training_epochs=FLAGS.training_epochs
    )
    model_save_dir = Path(FLAGS.architecture_dir).expanduser() / model_name

    if FLAGS.wandb:
        wandb.init(project=FLAGS.wandb_project)
        wandb.config.update(FLAGS)
        wandb.run.name = model_name + "_" + dt.datetime.now().strftime('%Y%m%d_%H%M%S')

    logging.info('Loading training data...')
    training_data, word_vocab, doc_vocab = _load_dataset_and_vocabs_from_file()
    training_data = training_data\
        .shuffle(buffer_size=FLAGS.batch_size * 5, reshuffle_each_iteration=True, seed=FLAGS.random_seed)\
        .batch(FLAGS.batch_size)\
        .prefetch(5)

    training_iter = training_data.as_numpy_iterator()

    # Choose 5 random words for qualitative evaluation through training
    comparison_terms = np.random.choice(word_vocab, size=5)

    logging.info('Initialising model...')
    model = hk.without_apply_rng(hk.transform(model_fn))

    if FLAGS.optimizer == 'sgd':
        optimizer = optax.sgd(
            learning_rate=FLAGS.learning_rate, momentum=FLAGS.momentum)
    elif FLAGS.optimizer == 'adam':
        optimizer = optax.adam(learning_rate=FLAGS.learning_rate)
    else:
        raise AssertionError('Expected optimizer `sgd` or `adam`')

    init_doc, init_context, _ = next(training_iter)
    model_params = model.init(jax.random.PRNGKey(FLAGS.random_seed), init_doc, init_context)
    opt_state = optimizer.init(model_params)

    logging.info('Beginning training...')
    for epoch in range(FLAGS.training_epochs):
        for b, batch in enumerate(training_iter):
            model_params, opt_state = update(model_params, opt_state, batch)

            if not b % FLAGS.log_every:
                accuracy = calc_accuracy(model_params, batch)
                loss = loss_fn(model_params, batch)
                accuracy, loss = jax.device_get(accuracy), jax.device_get(loss)

                logging.info(LOG_FMT_STRING.format(
                    epoch=epoch, batch=b, accuracy=accuracy, loss=loss))

                if FLAGS.wandb:
                    wandb.log({
                        'accuracy': accuracy,
                        'loss': loss,
                        'batch': batch,
                        'epoch': epoch
                    })

        # Save model at end of each epoch
        logging.info('Saving model to %s...', model_save_dir)
        _save_model_to(jax.device_get(model_params), model_save_dir)

        # Print similar terms
        _get_similar_terms(
            comparison_terms,
            word_vocab,
            model_params[FLAGS.architecture_name + '/~/word_embeddings']
        )

        training_iter = training_data.as_numpy_iterator()


if __name__ == '__main__':
    app.run(main)
