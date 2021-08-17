import os
import glob
from sklearn.utils import shuffle

def read_imdb_data(data_dir="/Users/anjalyparayil/PycharmProjects/NLP_Course/SentimentAnalysis/data"):
    data = {}
    labels = {}
    # sib dictionaries for train and test:
    for data_type in ['train', 'test']:
        data[data_type] = {}
        labels[data_type] = {}
        for sentiment in ['pos', 'neg']:
            data[data_type][sentiment] = []
            labels[data_type][sentiment] = []
            path = os.path.join(data_dir, data_type, sentiment, "*.txt")
            files = glob.glob(path)
            for f in files:
                with open(f) as review:
                    data[data_type][sentiment].append(review.read())
                    labels[data_type][sentiment].append(sentiment)
            assert len(data[data_type][sentiment]) == len(labels[data_type][sentiment]), \
                "{}/{} data size does not match labels size".format(data_type, sentiment)
    return data, labels

def prepare_imdb_data(data,labels):
    data_train = data['train']['pos'] + data['train']['neg']
    data_test = data['test']['pos'] + data['test']['neg']
    labels_train = labels['train']['pos'] + labels['train']['neg']
    labels_test = labels['test']['pos'] + labels['test']['neg']

    data_train, labels_train = shuffle(data_train, labels_train)
    data_test, labels_test = shuffle(data_test, labels_test)
    return data_train, data_test, labels_train, labels_test



