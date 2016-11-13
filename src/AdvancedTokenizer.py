from collections import defaultdict

import re
import nltk
from nltk.corpus import stopwords
from src.Document import Document
from src.Tokenizer import Tokenizer


class AdvancedTokenizer(Tokenizer):
    def __init__(self):
        self.tokenizer = nltk.tokenize.TreebankWordTokenizer()
        self.stopWords = set(stopwords.words('english'))
        self.regex = re.compile("[^\w\s]|_")

    def tokenize(self, document: Document):
        bow = defaultdict(float)
        content = document.getContent()
        tokens = self.tokenizer.tokenize(content)
        lowered_tokens = map(lambda t: self.scrubToken(t), tokens)
        for token in lowered_tokens:
            if token not in self.stopWords and token != '':
                bow[token] += 1.0
        return bow

    def scrubToken(self, token):
        scrubbedToken = self.regex.sub("", token)
        return scrubbedToken.lower()
