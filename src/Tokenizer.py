from src.Document import Document


class Tokenizer:
    '''
    This is an abstract tokenizer.
    '''
    def tokenize(self, document: Document):
        raise NotImplementedError("Abstract Tokenizer needs to be implemented")