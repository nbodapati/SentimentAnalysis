from collections import defaultdict

import nltk
from nltk.corpus import stopwords
from src.Document import Document
from src.Tokenizer import Tokenizer


class AdvancedTokenizer(Tokenizer):
    def __init__(self):
        self.tokenizer = nltk.tokenize.TreebankWordTokenizer()
        self.stopWords = set(stopwords.words('english'))

    def tokenize(self, document: Document):
        bow = defaultdict(float)
        content = document.getContent()
        tokens = self.tokenizer.tokenize(content)
        lowered_tokens = map(lambda t: t.lower(), tokens)
        for token in lowered_tokens:
            if token not in self.stopWords:
                bow[token] += 1.0
