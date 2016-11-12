from collections import defaultdict

import nltk
from src.Document import Document
from src.Tokenizer import Tokenizer


class BigramTokenizer(Tokenizer):
    def __init__(self):
        self.tokenizer = nltk.tokenize.TreebankWordTokenizer()

    def tokenize(self, document: Document):
        bow = defaultdict(float)
        content = document.getContent()
        tokens = self.tokenizer.tokenize(content)
        lowered_tokens = map(lambda t: t.lower(), tokens)

        bigrams = nltk.bigrams(lowered_tokens)
        for bigram in bigrams:
            bow[bigram] += 1.0
        return bow