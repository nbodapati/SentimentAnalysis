import os

from src.Document import Document


class SentenceDocument(Document):
    '''
    This is an implementation of the abstract document class to represent a document in the file system.
    '''
    def __init__(self, sentence):
        self.sentence = sentence

    def getContent(self):
       return self.sentence
