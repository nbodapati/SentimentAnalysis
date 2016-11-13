import codecs
import os

from src.AdvancedTokenizer import AdvancedTokenizer
from src.BigramTokenizer import BigramTokenizer
from src.FileDocument import FileDocument
from src.Labels import Labels
from src.NaiveBayesClassifier import NaiveBayesClassifier
from src.SentenceDocument import SentenceDocument
from src.Tokenizer import Tokenizer
from src.SimpleTokenizer import SimpleTokenizer
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

# Path to dataset
PATH_TO_POLARITY_DATA = '../../Datasets/review_polarity/txt_sentoken/'
PATH_TO_IMDB_TEST_DATA = '../../Datasets/aclImdb/test/'
PATH_TO_IMDB_TRAIN_DATA = '../../Datasets/aclImdb/train/'

PATH_TO_SUBJECTIVITY_DATA_SUBJECTIVE = '../../Datasets/rotten_imdb/quote.tok.gt9.5000'
PATH_TO_SUBJECTIVITY_DATA_OBJECTIVE = '../../Datasets/rotten_imdb/plot.tok.gt9.5000'

POS_LABEL = 'pos'
NEG_LABEL = 'neg'
K = 5


def getDocuments(label, path, fileNames):
    documents = []
    for fileName in fileNames:
        document = FileDocument(os.path.join(path, fileName))
        documents.append((document, label))
    return documents

def getSubjectivityDocuments(label, path):
    documents = []
    with open(path, "r", encoding='ISO-8859-1') as doc:
        content = doc.read()
        files = content.split('\n')
        for file in files:
            document = SentenceDocument(file)
            documents.append((document, label))
    return documents

def kFoldCrossValidate(k, classifier, training_set: np.array):
    kf = KFold(n_splits=k)

    for train_index, test_index in kf.split(training_set):
        training_data, test_data = training_set[train_index], training_set[test_index]
        classifier.train_model(training_data.tolist())


def evaluateReviewPolarity(k, tokenizer: Tokenizer, alphas):
    nb = NaiveBayesClassifier(tokenizer)

    pos_path = os.path.join(PATH_TO_POLARITY_DATA, POS_LABEL)
    neg_path = os.path.join(PATH_TO_POLARITY_DATA, NEG_LABEL)

    train_pos_documents = getDocuments(Labels.strong_pos, pos_path, os.listdir(pos_path)[:800])
    train_neg_documents = getDocuments(Labels.strong_neg, neg_path, os.listdir(neg_path)[:800])

    test_pos_documents = getDocuments(Labels.strong_pos, pos_path, os.listdir(pos_path)[800:])
    test_neg_documents = getDocuments(Labels.strong_neg, neg_path, os.listdir(neg_path)[800:])

    kFoldCrossValidate(k, nb, np.array(train_pos_documents + train_neg_documents))

    # printClassifierStatistics(nb)

    accuracies = []
    for alpha in alphas:
        accuracy = nb.evaluate(test_pos_documents + test_neg_documents, alpha)
        accuracies.append(accuracy)
    return accuracies


def evaluateIMDB(k, tokenizer: Tokenizer, alphas):
    nb = NaiveBayesClassifier(tokenizer)

    train_pos_path = os.path.join(PATH_TO_IMDB_TRAIN_DATA, POS_LABEL)
    train_neg_path = os.path.join(PATH_TO_IMDB_TRAIN_DATA, NEG_LABEL)
    train_pos_documents = getDocuments(Labels.strong_pos, train_pos_path, os.listdir(train_pos_path))
    train_neg_documents = getDocuments(Labels.strong_neg, train_neg_path, os.listdir(train_neg_path))

    test_pos_path = os.path.join(PATH_TO_IMDB_TEST_DATA, POS_LABEL)
    test_neg_path = os.path.join(PATH_TO_IMDB_TEST_DATA, NEG_LABEL)
    test_pos_documents = getDocuments(Labels.strong_pos, test_pos_path, os.listdir(test_pos_path))
    test_neg_documents = getDocuments(Labels.strong_neg, test_neg_path, os.listdir(test_neg_path))

    kFoldCrossValidate(k, nb, np.array(train_pos_documents + train_neg_documents))

    # printClassifierStatistics(nb)

    accuracies = []
    for alpha in alphas:
        accuracy = nb.evaluate(test_pos_documents + test_neg_documents, alpha)
        accuracies.append(accuracy)
    return accuracies


def evaluateSubjectivity(k, tokenizer: Tokenizer, alphas):
    nb = NaiveBayesClassifier(tokenizer)

    objective_documents = getSubjectivityDocuments(Labels.strong_neg, PATH_TO_SUBJECTIVITY_DATA_OBJECTIVE)
    subjective_documents = getSubjectivityDocuments(Labels.strong_pos, PATH_TO_SUBJECTIVITY_DATA_SUBJECTIVE)

    train_objective_documents = objective_documents[:4000]
    train_subjective_documents = subjective_documents[:4000]

    test_objective_documents = objective_documents[4000:]
    test_subjective_documents = subjective_documents[4000:]

    kFoldCrossValidate(k, nb, np.array(train_objective_documents + train_subjective_documents))

    # printClassifierStatistics(nb)

    accuracies = []
    for alpha in alphas:
        accuracy = nb.evaluate(test_objective_documents + test_subjective_documents, alpha)
        accuracies.append(accuracy)
    return accuracies


def plotAccuracies(tokenizerName, alphas, reviewPolarityAccuracies, imdbAccuracies):
    line_rp, = plt.plot(alphas, reviewPolarityAccuracies, 'r', label='Review Polarity')
    line_imdb, = plt.plot(alphas, imdbAccuracies, 'b', label='IMDB')
    plt.legend(handles=[line_rp, line_imdb], bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, borderaxespad=0.)
    fig1 = plt.gcf()
    plt.show()
    plt.draw()
    fig1.savefig("../results/{0}_Accuracy.png".format(tokenizerName.replace(" ", "_")))


def printTable(tokenizerName, alphas, reviewPolarityAccuracies, imdbAccuracies):
    print("###{0}".format(tokenizerName))
    print("| Alpha  | Review Polarity Accuracy | IMDB Accuracy |")
    print("|---|:---:|:---:|")
    for i in range(len(alphas)):
        print("| {0}  | {1}  | {2} |".format(alphas[i], reviewPolarityAccuracies[i], imdbAccuracies[i]))
    plotAccuracies(tokenizerName, alphas, reviewPolarityAccuracies, imdbAccuracies)


def printClassifierStatistics(classifier):
    classifier.report_statistics_after_training()
    print("Top 20 tokens in the positive class")
    for token in classifier.top_n(Labels.strong_pos, 20):
        print(token)

    print("Top 20 tokens in the negative class")
    for token in classifier.top_n(Labels.strong_neg, 20):
        print(token)


alphas = [5, 10, 15, 20, 25, 30, 35]
tokenizer = SimpleTokenizer()
# reviewPolarityAccuracies = evaluateReviewPolarity(K, tokenizer, alphas)
# imdbAccuracies = evaluateIMDB(K, tokenizer, alphas)
subjectivityAccuracies = evaluateSubjectivity(K, tokenizer, alphas)

# reviewPolarityAccuracies = [0.81, 0.83, 0.835, 0.8475, 0.8475, 0.85, 0.85, 0.845]
# imdbAccuracies = [0.82312, 0.83088, 0.83392, 0.83532, 0.83532, 0.83576, 0.83628, 0.83652]

# printTable("Simple Tokenizer", alphas, reviewPolarityAccuracies, imdbAccuracies)

# tokenizer = AdvancedTokenizer()
# reviewPolarityAccuracies = evaluateReviewPolarity(K, tokenizer, alphas)
# imdbAccuracies = evaluateIMDB(K, tokenizer, alphas)

# reviewPolarityAccuracies = [0.805, 0.8275, 0.8275, 0.83, 0.8425, 0.8325, 0.8425, 0.8275]
# imdbAccuracies = [0.83364, 0.84168, 0.8436, 0.84436, 0.84496, 0.84532, 0.84548, 0.84592]
# printTable("Advanced Tokenizer", alphas, reviewPolarityAccuracies, imdbAccuracies)

# tokenizer = BigramTokenizer()
# reviewPolarityAccuracies = evaluateReviewPolarity(K, tokenizer, alphas)
# imdbAccuracies = evaluateIMDB(K, tokenizer, alphas)

# reviewPolarityAccuracies = [0.805, 0.8275, 0.8275, 0.83, 0.8425, 0.8325, 0.8425, 0.8275]
# imdbAccuracies = [0.83364, 0.84168, 0.8436, 0.84436, 0.84496, 0.84532, 0.84548, 0.84592]
# printTable("Bigram Tokenizer", alphas, reviewPolarityAccuracies, imdbAccuracies)

print(subjectivityAccuracies)