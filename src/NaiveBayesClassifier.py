"""

"""
from collections import defaultdict

import math
from operator import itemgetter

from src.Labels import Labels


class NaiveBayesClassifier():
    '''
    A Naive Bayes Classifier.
    '''

    def __init__(self, tokenizer, alpha: object = 0.0) -> object:
        self.vocabulary = set()
        self.document_counts = defaultdict(float)
        self.total_document_count = 0.0
        self.word_counts = defaultdict(lambda : defaultdict(float))
        self.total_word_counts = defaultdict(float)
        self.tokenizer = tokenizer
        self.alpha = alpha
        self.labels = set()

    def train_model(self, training_data):
        for example, label in training_data:
            self.labels.add(label)
            self.total_document_count += 1
            self.tokenize_and_update_model(example, label)

    def tokenize_and_update_model(self, document, label):
        bow = self.tokenizer(document)
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

    def log_likelihood(self, bow, label):
        likelihood = 0.0
        for token in bow:
            likelihood += math.log(self.word_counts[label][token] + self.alpha) - math.log(self.total_word_counts[label] + self.alpha * len(self.vocabulary))
        return likelihood

    def log_posterior(self, bow, label):
        return self.log_likelihood(bow, label) + self.log_prior(label)

    def classify(self, bow):
        scores = [(label, self.log_posterior(bow, label)) for label in self.labels]
        class_label = max(scores, key=itemgetter(1))[0]
        return class_label

    def evaluate(self, test_data):
        tp = tn = fp = fn = 0.0

        for document, label in test_data:
            bow = self.tokenizer(document)
            classification = self.classify(bow)

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
