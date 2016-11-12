import unittest
from collections import defaultdict
from unittest.mock import Mock

import src.NaiveBayesClassifier
from src.Labels import Labels
from src.SimpleTokenizer import SimpleTokenizer


class NaiveBayesClassifierTests(unittest.TestCase):

    def setUp(self):
        self.alpha = 0.65
        self.tokenizer = SimpleTokenizer()
        self.nb = src.NaiveBayesClassifier.NaiveBayesClassifier(self.tokenizer)

    def test_init(self):
        self.assertEqual(self.nb.vocabulary, set())
        self.assertDictEqual(self.nb.document_counts, defaultdict(float))
        self.assertEqual(self.nb.total_document_count, 0.0)
        self.assertDictEqual(self.nb.word_counts, defaultdict( lambda : defaultdict(float)))
        self.assertDictEqual(self.nb.total_word_counts, defaultdict(float))
        self.assertEqual(self.nb.tokenizer, self.tokenizer)

    def test_train_model(self):
        training_data = []

        mockDocument1 = Mock()
        mockDocument1.getContent.return_value ="This is a great movie"
        training_data.append( (mockDocument1, Labels.strong_pos))

        mockDocument2 = Mock()
        mockDocument2.getContent.return_value = "This is a bad movie"
        training_data.append( (mockDocument2, Labels.strong_neg))

        mockDocument3 = Mock()
        mockDocument3.getContent.return_value = "This is a nice movie"
        training_data.append( (mockDocument3, Labels.strong_pos))

        mockDocument4 = Mock()
        mockDocument4.getContent.return_value = "this is an awful movie"
        training_data.append( (mockDocument4, Labels.strong_neg))

        self.nb.train_model(training_data)
        self.assertEqual(self.nb.total_document_count, 4)
        self.assertDictEqual(self.nb.word_counts[Labels.strong_pos], {"this": 2.0, "is": 2.0, "a": 2.0, "great": 1.0, "movie": 2.0, "nice": 1.0})
        self.assertDictEqual(self.nb.word_counts[Labels.strong_neg],
                     {"this": 2.0, "is": 2.0, "a": 1.0, "an": 1.0, "bad": 1.0, "movie": 2.0, "awful": 1.0})
        self.assertEqual(self.nb.total_word_counts[Labels.strong_pos], 10.0)
        self.assertEqual(self.nb.total_word_counts[Labels.strong_neg], 10.0)

    def test_classify(self):
        training_data = []

        mockDocument1 = Mock()
        mockDocument1.getContent.return_value ="This is a great movie"
        training_data.append( (mockDocument1, Labels.strong_pos))

        mockDocument2 = Mock()
        mockDocument2.getContent.return_value = "This is a bad movie"
        training_data.append( (mockDocument2, Labels.strong_neg))

        mockDocument3 = Mock()
        mockDocument3.getContent.return_value = "This is a nice movie"
        training_data.append( (mockDocument3, Labels.strong_pos))

        mockDocument4 = Mock()
        mockDocument4.getContent.return_value = "this is an awful movie"
        training_data.append( (mockDocument4, Labels.strong_neg))

        self.nb.train_model(training_data)

        mockDocument = Mock()
        mockDocument.getContent.return_value = "This is great"
        bow = self.tokenizer.tokenize(mockDocument)
        label = self.nb.classify(bow, self.alpha)
        self.assertEqual(label, Labels.strong_pos)

    def test_evaluate(self):
        training_data = []

        mockDocument1 = Mock()
        mockDocument1.getContent.return_value ="This is a great movie"
        training_data.append( (mockDocument1, Labels.strong_pos))

        mockDocument2 = Mock()
        mockDocument2.getContent.return_value = "This is a bad movie"
        training_data.append( (mockDocument2, Labels.strong_neg))

        mockDocument3 = Mock()
        mockDocument3.getContent.return_value = "This is a nice movie"
        training_data.append( (mockDocument3, Labels.strong_pos))

        mockDocument4 = Mock()
        mockDocument4.getContent.return_value = "this is an awful movie"
        training_data.append( (mockDocument4, Labels.strong_neg))

        mockDocument5 = Mock()
        mockDocument5.getContent.return_value = "The acting is so bad"
        training_data.append( (mockDocument5, Labels.strong_neg))

        mockDocument6 = Mock()
        mockDocument6.getContent.return_value = "the direction is great"
        training_data.append( (mockDocument6, Labels.strong_neg))

        self.nb.train_model(training_data)

        test_data = []

        mockDocument1 = Mock()
        mockDocument1.getContent.return_value = "This is not so bad"
        test_data.append( (mockDocument1, Labels.strong_pos))

        mockDocument2 = Mock()
        mockDocument2.getContent.return_value = "This is nice"
        test_data.append( (mockDocument2, Labels.strong_pos))

        mockDocument3 = Mock()
        mockDocument3.getContent.return_value = "This is awful"
        test_data.append( (mockDocument3, Labels.strong_neg))

        mockDocument4 = Mock()
        mockDocument4.getContent.return_value = "this is not good"
        test_data.append( (mockDocument4, Labels.strong_neg))

        mockDocument5 = Mock()
        mockDocument5.getContent.return_value = "this is a great movie"
        test_data.append( (mockDocument5, Labels.strong_pos))

        accuracy = self.nb.evaluate(test_data, self.alpha)
        self.assertEqual(accuracy, 0.60)
