"""Text helper functions for Doc2Vec training."""
import collections
import itertools
from typing import Collection, Dict, List, Optional


def flatten_nested_text(nested_text: List[List[str]]) -> List[str]:
    return list(itertools.chain.from_iterable(nested_text))


class Vocabulary:
    """Simple object to assign and return unique ids to words.

    Usage: if max_size is set, first fit to corpus using .fit().
    Return word ids like vocab[word].
    """

    def __init__(self, max_size: Optional[int] = None, unk_token: str = '[UNK]'):
        """Initialise Vocab object.

        Args:
            max_size: constrain vocabulary to given size. If None, will return an
            incremental id for each new word passed. Otherwise it will assign ids to
            the max_size most common words upon calling .fit(), with new or
            low-frequency words returning id 0.
        """
        self.vocab = {unk_token: 0}
        self.max_size = max_size
        self.counter = None
        self.total_words = 0

    def __getitem__(self, token: str) -> int:
        if not self.max_size:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab) + 1

            return self.vocab[token]

        else:
            if not self.counter:
                raise ValueError('max_size is set. Call .fit() to fit vocab.')
            elif token not in self.vocab:
                return 0
            else:
                return self.vocab[token]

    def __len__(self):
        return len(self.vocab)

    def fit(self, words: Collection[str]) -> Dict[str, int]:
        if not self.max_size:
            raise ValueError('max_size must be set before calling .fit().')

        self.counter = collections.Counter(words)
        for word, word_freq in self.counter.most_common(self.max_size):
            self.vocab[word] = len(self.vocab) + 1
            self.total_words += word_freq

        return self.vocab
