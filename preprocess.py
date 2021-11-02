from __future__ import division
from __future__ import print_function

import re
import numpy as np
import six
from tensorflow.contrib import learn
from tensorflow.python.platform import gfile
from tensorflow.contrib import learn  # pylint: disable=g-bad-import-order

TOKENIZER_RE = re.compile(r"[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+",
                          re.UNICODE)

class MyVocabularyProcessor(learn.preprocessing.VocabularyProcessor):
    def __init__(self,
               max_seq_len,
               max_word_len,
               min_frequency=0,
               vocabulary=None,
               alphabet_id=None
               ):
        self.max_word_len = max_word_len
        self.alphabet_id = alphabet_id
        sup = super(MyVocabularyProcessor,self)
        sup.__init__(max_seq_len,min_frequency, vocabulary)
        self.max_seq_len = max_seq_len
        self.max_word_len = max_word_len
        self.alphabet_id = alphabet_id

    def padSequence(self, raw_document, mode):
        """Transform documents to word-id matrix.
        Convert words to ids with vocabulary fitted with fit or the one
        provided in the constructor.
        Args:
          raw_documents: An iterable which yield either str or unicode.
        Yields:
          x: iterable, [n_samples, max_document_length]. Word-id matrix.
        """
        length = self.max_seq_len
        for tokens in self._tokenizer(raw_document):
            word_ids = np.zeros(length, np.int64)
            for idx, token in enumerate(tokens):
                if idx >= length:
                    break
                word_ids[idx] = self.vocabulary_.get(token)
            yield tokens, word_ids

    def padChar(self, raw_document):
        for tokens in self._tokenizer(raw_document):
            char_ids = []
            for i in range(self.max_seq_len):
                char_ids.append(np.zeros(self.max_word_len, np.int64))
            seq_len = len(tokens)
            for i in range(min(seq_len, self.max_seq_len)):
                word_len = len(tokens[i])
                for j in range(min(word_len, self.max_word_len)):
                    if tokens[i][j] not in self.alphabet_id.keys():
                        char_ids[i][j] = 0
                    else:
                        char_ids[i][j] = self.alphabet_id[tokens[i][j]]
            yield char_ids
