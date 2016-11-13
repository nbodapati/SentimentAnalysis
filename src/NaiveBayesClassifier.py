"""

"""
from collections import defaultdict

import math
from operator import itemgetter

from src.Document import Document
from src.Labels import Labels
from src.Tokenizer import Tokenizer


class NaiveBayesClassifier():
    '''
    A Naive Bayes Classifier.
    '''

    def __init__(self, tokenizer: Tokenizer) -> object:
        self.vocabulary = set()
        self.document_counts = defaultdict(float)
        self.total_document_count = 0.0
        self.word_counts = defaultdict(lambda : defaultdict(float))
        self.total_word_counts = defaultdict(float)
        self.tokenizer = tokenizer
        self.labels = set()

    def train_model(self, training_data):
        for document, label in training_data:
            self.labels.add(label)
            self.total_document_count += 1
            self.tokenize_and_update_model(document, label)

    def tokenize_and_update_model(self, document: Document, label):
        bow = self.tokenizer.tokenize(document)
        self.update_model(bow, label)

    def update_model(self, bow, label):
        self.document_counts[label] += 1.0
        for word in bow:
            self.vocabulary.add(word)
            self.word_counts[label][word] += bow[word]
            self.total_word_counts[label] += bow[word]

    def log_prior(self, label):
        '''
        :param label: The label
        :return: log prior for the label
        '''
        return math.log(self.document_counts[label]) - math.log(self.total_document_count)

    def log_likelihood(self, bow, label, alpha):
        likelihood = 0.0
        for token in bow:
            likelihood += math.log(self.word_counts[label][token] + alpha) - math.log(self.total_word_counts[label] + alpha * len(self.vocabulary))
        return likelihood

    def log_posterior(self, bow, label, alpha):
        return self.log_likelihood(bow, label, alpha) + self.log_prior(label)

    def classify(self, bow, alpha):
        scores = [(label, self.log_posterior(bow, label, alpha)) for label in self.labels]
        class_label = max(scores, key=itemgetter(1))[0]
        return class_label

    def report_statistics_after_training(self):
        """
        Report a number of statistics after training.
        """

        print("Vocabulary Size: Number of unique word types in training corpus:", len(self.vocabulary))
        for label in self.labels:
            print("Number of tokens in {0} class: {1}".format(label, len(self.word_counts[label])))

    def top_n(self, label, n):
        """

        Returns the most frequent n tokens for documents with class 'label'.
        """
        return sorted(self.word_counts[label].items(), key=lambda item: -item[1])[:n]


    def evaluate(self, test_data, alpha):
        tp = tn = fp = fn = 0.0

        for document, label in test_data:
            bow = self.tokenizer.tokenize(document)
            classification = self.classify(bow, alpha)

            if classification == Labels.strong_pos and label == Labels.strong_pos:
                tp += 1.0
            elif classification == Labels.strong_pos and label == Labels.strong_neg:
                fp += 1.0
            elif classification == Labels.strong_neg and label == Labels.strong_neg:
                tn += 1.0
            elif classification == Labels.strong_neg and label == Labels.strong_pos:
                fn += 1.0

        accuracy = (tn + tp) / (tp + tn + fn + fp)
        return accuracy
