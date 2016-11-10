import os

from src.FileDocument import FileDocument
from src.Labels import Labels
from src.NaiveBayesClassifier import NaiveBayesClassifier
from src.SimpleTokenizer import SimpleTokenizer
import numpy as np
from sklearn.model_selection import KFold




def getDocuments(label, path, fileNames):
    documents = []
    for fileName in fileNames:
        document = FileDocument(path, fileName)
        documents.append((document, label))
    return documents


# Path to dataset
PATH_TO_POLARITY_DATA = '../review_polarity/txt_sentoken/'

POS_LABEL = 'pos'
NEG_LABEL = 'neg'

kf = KFold(n_splits=5)

tokenizer = SimpleTokenizer()

pos_path = os.path.join(PATH_TO_POLARITY_DATA, POS_LABEL)
neg_path = os.path.join(PATH_TO_POLARITY_DATA, NEG_LABEL)

pos_files = np.array(os.listdir(pos_path))
neg_files = np.array(os.listdir(neg_path))

nb = NaiveBayesClassifier(tokenizer, alpha=1.0)

for train_index, test_index in kf.split(pos_files):
    train_data_pos, test_data_pos = pos_files[train_index], pos_files[test_index]
    train_data_neg, test_data_neg = neg_files[train_index], neg_files[test_index]

    train_documents_pos = getDocuments(Labels.strong_pos, pos_path, train_data_pos)
    train_documents_neg = getDocuments(Labels.strong_neg, neg_path, train_data_neg)

    nb.train_model(train_documents_neg + train_documents_pos)

    test_documents_pos = getDocuments(Labels.strong_pos, pos_path, test_data_pos)
    test_documents_neg = getDocuments(Labels.strong_neg, neg_path, test_data_neg)

    accuracy = nb.evaluate(test_documents_neg + test_documents_pos)
    print(accuracy)
