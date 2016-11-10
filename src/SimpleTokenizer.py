from collections import defaultdict

from src.Document import Document
from src.Tokenizer import Tokenizer


class SimpleTokenizer(Tokenizer):

    def tokenize(self, document: Document):
        bow = defaultdict(float)
        content = document.getContent()
        tokens = content.split()
        lowered_tokens = map(lambda t: t.lower(), tokens)
        for token in lowered_tokens:
            bow[token] += 1.0
        return bow