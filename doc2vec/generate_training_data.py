"""Pre-process IMDB reviews for training.

Note that this is implemented without the help of higher-level libraries like
tf.Data or Beam. This has major performance implications -- notably, the dataset
must fit in memory -- but is intended to improve instructive value.
"""
from absl import app
from absl import flags
from absl import logging

import hashlib
from pathlib import Path
import re
from typing import List, Tuple

import numpy as np

from doc2vec.text_helpers import flatten_nested_text, Vocabulary

FLAGS = flags.FLAGS
flags.DEFINE_integer('vocab_size', default=50_000, help='Vocab size')
flags.DEFINE_integer('window_size', default=5, help='Window size')
flags.DEFINE_string('training_data_dir', default=None,
                    help='Path to directory containing training data. We expect a file for each document in the '
                         'directory (corpus). Prepped data will be written to a sub-directory.')
flags.DEFINE_string('dataset_name', None, help='Identifier for this dataset')

flags.register_validator(
    flag_name='training_data_dir',
    checker=lambda fp: Path(fp).expanduser().exists(),
    message='One of your file/directory paths appears not to exist'
)

NP_DTYPE = np.uint32
DATASET_NAME_PATTERN = 'doc2vec_pvdm_{dataset_name}_{window_size}window_{vocab_size}vocab'


def standardize_text(raw_document: str) -> List[str]:
    """Normalise text.

    Inspired by: https://www.tensorflow.org/tutorials/text/word_embeddings#text_preprocessing
    """
    output = raw_document.lower()  # lowercase
    output = re.sub(r'<br />', ' ', output)  # remove HTML line breaks
    # TODO: fix this so punctuation doens't generate unnecessary spaces
    output = re.sub(r'[\W]+', ' ', output)  # remove non-alphanumerics
    output = re.sub(r'\s{2,}', ' ', output).strip()  # remove multiple spaces
    return output.split()  # tokenize on whitespace


def _hash_text(raw_document: str) -> str:
    """Return MD5 representation of a document."""
    return hashlib.sha256(raw_document.encode('utf-8')).hexdigest()


def _encode_text(document: List[str], vocab: Vocabulary) -> List[int]:
    """Integer encodes words in a document with provided vocabulary."""
    return [vocab[word] for word in document]


def _build_training_examples(text: List[int], doc_id: int, window_size: int) -> List[Tuple[int, np.ndarray, int]]:
    examples = []

    # Take a sliding window of length `window_size` to RHS of target word.
    for w_idx in range(1, len(text) - window_size):
        target_word = text[w_idx - 1]
        context_words = text[w_idx:w_idx + window_size]

        examples.append((
            doc_id,
            np.array(context_words, dtype=NP_DTYPE),
            target_word
        ))

    return examples


def run_pipeline(unused_argv):
    """Prepare raw text data for training.

    Sequentially, we:
    1. Load documents into memory
    2. Create a hash of each document to serve as a unique identifier
    3. Clean and tokenize text
    4. Create vocabularies for words and documents
    5. Integer encode documents
    6. Prepare training examples
    7. Save results to disk (in .npy format)
    """

    # === 1. Read all training documents into memory ===
    logging.info('Reading documents from file...')

    raw_documents = []
    for document_fp in Path(FLAGS.training_data_dir).expanduser().iterdir():
        if document_fp.is_dir():
            continue

        with open(document_fp, 'r') as d:
            raw_documents.append(d.read())

    # === 2. Hash each document to serve as unique identifier ===
    logging.info('Hashing documents...')

    document_ids = [_hash_text(raw_doc) for raw_doc in raw_documents]

    # === 3. Clean and tokenize text ===
    logging.info('Standardizing and tokenizing documents...')

    clean_documents = [standardize_text(document) for document in raw_documents]

    # Drop raw text from memory as we no longer need them
    del raw_documents

    # === 4. Fit word and document vocabularies ===
    logging.info('Building vocabularies...')

    word_vocab = Vocabulary(max_size=FLAGS.vocab_size)
    word_vocab.fit(flatten_nested_text(clean_documents))

    doc_vocab = Vocabulary()  # no max size - this will give iterative idxs

    # === 5. Integer encode document text and ids ===
    logging.info('Integer encoding documents...')

    encoded_documents = [_encode_text(doc, word_vocab) for doc in clean_documents]
    encoded_doc_ids = [doc_vocab[doc_id] for doc_id in document_ids]

    # We can now also drop clean_documents
    del clean_documents

    # === 6. Prepare training examples ===
    logging.info('Preparing training examples...')

    zipped_docs_and_ids = zip(encoded_documents, encoded_doc_ids)

    training_examples = []
    for doc, doc_id in zipped_docs_and_ids:
        training_examples.extend(
            _build_training_examples(doc, doc_id, FLAGS.window_size)
        )

    del zipped_docs_and_ids

    doc_ids, context_words, target_words = zip(*training_examples)

    doc_ids = np.array(doc_ids, dtype=NP_DTYPE)
    context_words = np.array(context_words, dtype=NP_DTYPE)
    target_words = np.array(target_words, dtype=NP_DTYPE)

    # === 7. Save training data and vocabularies to file ===
    logging.info('Saving data to file')

    out_dir = Path(FLAGS.training_data_dir).expanduser() / 'training_data' / \
        DATASET_NAME_PATTERN.format(
            dataset_name=FLAGS.dataset_name,
            window_size=FLAGS.window_size,
            vocab_size=FLAGS.vocab_size
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save training examples to NPY
    np.save(out_dir / 'doc_ids', doc_ids)
    np.save(out_dir / 'context_words', context_words)
    np.save(out_dir / 'target_words', target_words)

    # Save vocabs
    with open(out_dir / 'word_vocab.txt', 'w') as f:
        f.write('\n'.join(word_vocab.vocab.keys()))

    with open(out_dir / 'doc_vocab.txt', 'w') as f:
        f.write('\n'.join(doc_vocab.vocab.keys()))


if __name__ == '__main__':
    app.run(run_pipeline)
