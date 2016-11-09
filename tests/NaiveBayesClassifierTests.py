import unittest
import unittest.mock
from collections import defaultdict

import src.NaiveBayesClassifier
from src.Labels import Labels


class NaiveBayesClassifierTests(unittest.TestCase):

    @staticmethod
    def tokenizer(document):
        bow = defaultdict(float)
        tokens = document.split()
        lowered_tokens = map(lambda t: t.lower(), tokens)
        for token in lowered_tokens:
            bow[token] += 1.0
        return bow

    def setUp(self):
        self.alpha = 0.65
        self.nb = src.NaiveBayesClassifier.NaiveBayesClassifier(self.tokenizer, self.alpha)

    def test_init(self):
        self.assertEqual(self.nb.vocabulary, set())
        self.assertDictEqual(self.nb.document_counts, defaultdict(float))
        self.assertEqual(self.nb.total_document_count, 0.0)
        self.assertDictEqual(self.nb.word_counts, defaultdict( lambda : defaultdict(float)))
        self.assertDictEqual(self.nb.total_word_counts, defaultdict(float))
        self.assertEqual(self.nb.alpha, self.alpha)
        self.assertEqual(self.nb.tokenizer, self.tokenizer)

    def test_train_model(self):
        training_data = [("This is a great movie", Labels.strong_pos), ("This is a bad movie", Labels.strong_neg),
                         ("This is a nice movie", Labels.strong_pos), ("this is an awful movie", Labels.strong_neg)]
        self.nb.train_model(training_data)
        self.assertEqual(self.nb.total_document_count, 4)
        self.assertDictEqual(self.nb.word_counts[Labels.strong_pos], {"this": 2.0, "is": 2.0, "a": 2.0, "great": 1.0, "movie": 2.0, "nice": 1.0})
        self.assertDictEqual(self.nb.word_counts[Labels.strong_neg],
                     {"this": 2.0, "is": 2.0, "a": 1.0, "an": 1.0, "bad": 1.0, "movie": 2.0, "awful": 1.0})
        self.assertEqual(self.nb.total_word_counts[Labels.strong_pos], 10.0)
        self.assertEqual(self.nb.total_word_counts[Labels.strong_neg], 10.0)

    def test_classify(self):
        training_data = [("This is a great movie", Labels.strong_pos), ("This is a bad movie", Labels.strong_neg),
                         ("This is a nice movie", Labels.strong_pos), ("this is an awful movie", Labels.strong_neg)]
        self.nb.train_model(training_data)

        bow = self.tokenizer("This is great")
        label = self.nb.classify(bow)
        self.assertEqual(label, Labels.strong_pos)

    def test_evaluate(self):
        training_data = [("This is a great movie", Labels.strong_pos), ("This is a bad movie", Labels.strong_neg),
                         ("The acting is so bad", Labels.strong_neg),("This is a nice movie", Labels.strong_pos),
                         ("this is an awful movie", Labels.strong_neg), ("the direction is great", Labels.strong_pos)]
        self.nb.train_model(training_data)

        test_data = [("This is not so bad", Labels.strong_pos), ("This is nice", Labels.strong_pos),
                     ("This is awful", Labels.strong_neg), ("this is not good", Labels.strong_neg),
                     ("this is a great movie", Labels.strong_pos)]
        accuracy = self.nb.evaluate(test_data)
        self.assertEqual(accuracy, 0.60)
