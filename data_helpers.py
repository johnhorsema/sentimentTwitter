import numpy as np
import re
import itertools
from collections import Counter


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data(test_data_file):
    # Load data from files
    raw = list(open(test_data_file, "r").readlines()[1:])
    x_text = []
    for l in raw:
        if len(l.split("\t"))<3:
            x_text.append("")
        else:
            x_text.append(clean_str(l.split("\t")[2].strip()))
    y_labels = [0 for l in raw]
    return [np.array(x_text), np.array(y_labels)]

def load_data_and_labels(train_data_file):
    """
    Loads Kaggle Rotten Tomatoes data from file.
    Returns split sentences and labels, sorted by labels.
    """
    # Load data from files
    raw = list(open(train_data_file, "r").readlines()[1:])
    sentence_score_pairs = [(clean_str(l.split("\t")[2].strip()),int(l.split("\t")[3].strip())) for l in raw]
    x_text = [x for x,y in sentence_score_pairs]
    one_hots = np.eye(len(set([y for x,y in sentence_score_pairs])), dtype = int)
    y_labels = [np.array(one_hots[y]) for x,y in sentence_score_pairs]
    return [np.array(x_text), np.array(y_labels)]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            np.random.shuffle(data)
            shuffled_data = data
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]