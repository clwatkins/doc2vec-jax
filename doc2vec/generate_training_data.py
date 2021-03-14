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
from typing import List, Tuple, Optional, Sequence
from tqdm import tqdm
import numpy as np

from doc2vec.text_helpers import flatten_nested_text, Vocabulary

FLAGS = flags.FLAGS
flags.DEFINE_enum('architecture', 'dbow', ['pvdm', 'dbow'],
                  help='The model variant to select.')
flags.DEFINE_integer('vocab_size', default=50_000, help='Vocab size')
flags.DEFINE_integer('window_size', default=5, help='Window size')
flags.DEFINE_float('subsampling_thresh', default=10e-5,
                   help='Sub-sampling threshold (used to compute probability of discarding common words '
                        'in training data)')
flags.DEFINE_integer('ns_ratio', 4, help='Number of negative/noise words to sample for each positive. '
                                         'If 0 won\'t use negative sampling')
flags.DEFINE_string('training_data_dir', default=None,
                    help='Path to directory containing training data. We expect a file for each document in the '
                         'directory (corpus). Prepped data will be written to a sub-directory.')
flags.DEFINE_string('dataset_name', None, help='Identifier for this dataset')
flags.DEFINE_integer('random_seed', 1234, help='Seed used to shuffle input data and initialise model')

flags.register_validator(
    flag_name='training_data_dir',
    checker=lambda fp: Path(fp).expanduser().exists(),
    message='One of your file/directory paths appears not to exist'
)

NP_DTYPE = np.uint32
DATASET_NAME_PATTERN = 'doc2vec_{architecture}_{dataset_name}_{window_size}window_{vocab_size}' \
                       'vocab_{subsampling_thresh}subsampling_thresh{ns_ratio}nsratio'


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


def _build_training_examples(text: Sequence[int], doc_id: int,
                             window_size: int,
                             word_vocab: Vocabulary,
                             ns_sampling_table: Optional[np.ndarray],
                             subsampling_discard_probs: Optional[np.ndarray]
                             ) -> Tuple[List[int], List[np.ndarray], List[int], List[int]]:
    """Build Doc2Vec training examples for a given document.

    This function is architecture aware, including context words only for the PV-DM Doc2Vec variant. It optionally
    will build negative examples from the noise distribution and sub-sample common words -- controlled by the `num_ns`
    and `subsampling_thresh` params.

    Args:
        text: document, represented by sequence of integer-encoded words
        doc_id: unique document identifier, assumed integer-encoded
        window_size: width of one-sided window that is used to select context words for the PV-DM model
        word_vocab: word vocabulary
        ns_sampling_table: Optionally, an array of size (len(word_vocab),) that is used to weigh a word's probability
        of being selected as a negative sample
        subsampling_discard_probs: Optionally, an array of size (len(word_vocab),) that represents a word's probability
        of being discarded as a target word, based on its relative over-representation in the corpus

    Returns:
        training examples in their constituent parts -- doc ids, context words, target words, and labels
    """
    doc_ids, contexts, targets, labels = [], [], [], []
    vocab_array = np.array(list(word_vocab.vocab.values()))

    # Take a sliding window of length `window_size` to RHS of target word.
    for w_idx in range(1, len(text) - window_size):
        # === Positive labels ===
        target_word = text[w_idx - 1]

        # If sub-sampling, sample from the discard probabilities table, choosing whether to skip the target_word
        # Note that the UNK token (mapped to all words outside of the vocabulary's range) is assigned probability 0
        # and so will always be discarded
        if subsampling_discard_probs is not None \
                and np.random.uniform() < subsampling_discard_probs[target_word]:
            continue

        if FLAGS.architecture == 'pvdm':
            context_words = text[w_idx:w_idx + window_size]
        else:
            # DBOW doesn't require context words -- use a placerholder so the data struct can stay consistent
            # Because we eventually cast to a np.uint, this will end up being written out as the max value of the type
            context_words = [-1]

        # === Negative labels ===
        # For each positive example (of a true target_word) draw
        # ns_ratio 'noise' words from the underlying vocab distribution
        if FLAGS.ns_ratio:
            noise_words = np.random.choice(
                a=vocab_array,
                p=ns_sampling_table,
                size=FLAGS.ns_ratio
            ).tolist()
        else:
            noise_words = []

        doc_ids.extend([doc_id]*(1+FLAGS.ns_ratio))
        contexts.extend(context_words*(1+FLAGS.ns_ratio))
        targets.extend([target_word] + noise_words)
        labels.extend([1] + [0]*FLAGS.ns_ratio)

    return doc_ids, contexts, targets, labels


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

    # Set random seed for sampling reproducibility
    np.random.seed(FLAGS.random_seed)

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

    # Note that UNK (the first word in the vocab) doesn't have a corresponding Counter entry -- will return 0
    word_counts = np.array([word_vocab.counter[word] for word in word_vocab.vocab.keys()])

    if FLAGS.ns_ratio:
        # Raise observed word frequencies and total count to the 3/4 power as suggested in the Word2Vec paper (2.2)
        # https://proceedings.neurips.cc/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf
        #
        # This 'smooths' the distribution. Nice analysis by Chris McCormick:
        # http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/
        ns_sampling_probs = np.power(word_counts, 0.75) / np.power(word_vocab.total_words, 0.75)
        ns_sampling_probs = ns_sampling_probs / np.sum(ns_sampling_probs)
    else:
        ns_sampling_probs = None

    if FLAGS.subsampling_thresh:
        # In other implementations (like Keras') these probabilities are calculated
        # assuming a Zipf-like distribution in the underlying data.
        # Here we compute it empirically based on observed frequency
        # Rare values can end up with negative discard probs (-inf for 0 probability terms)
        # So clip to 0. Cf: https://stackoverflow.com/a/58773864
        word_probs = word_counts / word_vocab.total_words

        # Ignore divide-by-zero as it'll assign -inf and then be clipped
        with np.errstate(divide='ignore'):
            subsampling_discard_probs = 1 - np.sqrt(FLAGS.subsampling_thresh / word_probs)
        subsampling_discard_probs = np.clip(subsampling_discard_probs, a_min=0, a_max=None)
    else:
        subsampling_discard_probs = None

    doc_ids, contexts, targets, labels = [], [], [], []

    for doc, doc_id in tqdm(zipped_docs_and_ids, total=len(encoded_documents)):
        _doc_ids, _contexts, _targets, _labels = _build_training_examples(
            doc, doc_id, FLAGS.window_size, word_vocab, ns_sampling_probs, subsampling_discard_probs)

        doc_ids.extend(_doc_ids)
        contexts.extend(_contexts)
        targets.extend(_targets)
        labels.extend(_labels)

    del zipped_docs_and_ids

    doc_ids = np.array(doc_ids, dtype=NP_DTYPE)
    target_words = np.array(targets, dtype=NP_DTYPE)
    context_words = np.array(contexts, dtype=NP_DTYPE)
    labels = np.array(labels, dtype=NP_DTYPE)

    # === 7. Save training data and vocabularies to file ===
    logging.info('Saving data to file')

    out_dir = Path(FLAGS.training_data_dir).expanduser() / 'training_data' / \
        DATASET_NAME_PATTERN.format(
            dataset_name=FLAGS.dataset_name,
            architecture=FLAGS.architecture,
            window_size=FLAGS.window_size,
            vocab_size=FLAGS.vocab_size,
            subsampling_thresh=FLAGS.subsampling_thresh,
            ns_ratio=FLAGS.ns_ratio
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save training examples to NPY
    np.save(out_dir / 'doc_ids', doc_ids)
    np.save(out_dir / 'target_words', target_words)
    np.save(out_dir / 'context_words', context_words)
    np.save(out_dir / 'labels', labels)

    # Save vocabs
    with open(out_dir / 'word_vocab.txt', 'w') as f:
        f.write('\n'.join(word_vocab.vocab.keys()))

    with open(out_dir / 'doc_vocab.txt', 'w') as f:
        f.write('\n'.join(doc_vocab.vocab.keys()))


if __name__ == '__main__':
    app.run(run_pipeline)
