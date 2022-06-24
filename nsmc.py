# -*- coding: utf-8 -*-
import numpy as np
import hgtk
import pandas as pd
import io, os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from subchar_rule import subchar_dict

import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Embedding, Conv1D, LSTM, Dropout
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K

END_CHAR = "&"
END_JAMO = "^"

tf.get_logger().setLevel('ERROR')

def recall(y_target, y_pred):
    y_target_yn = K.round(K.clip(y_target, 0, 1))
    y_pred_yn = K.round(K.clip(y_pred, 0, 1))

    count_true_positive = K.sum(y_target_yn * y_pred_yn)

    count_true_positive_false_negative = K.sum(y_target_yn)

    recall = count_true_positive / (count_true_positive_false_negative + K.epsilon())

    return recall


def precision(y_target, y_pred):
    y_pred_yn = K.round(K.clip(y_pred, 0, 1))  
    y_target_yn = K.round(K.clip(y_target, 0, 1)) 

    count_true_positive = K.sum(y_target_yn * y_pred_yn)

    count_true_positive_false_positive = K.sum(y_pred_yn)

    precision = count_true_positive / (count_true_positive_false_positive + K.epsilon())

    return precision


def f1score(y_target, y_pred):
    _recall = recall(y_target, y_pred)
    _precision = precision(y_target, y_pred)
    _f1score = (2 * _recall * _precision) / (_recall + _precision + K.epsilon())

    return _f1score


def main():
    train_code = sys.argv[1]
    vocab = sys.argv[2]
    epochs = eval(sys.argv[3])
    dropout = eval(sys.argv[4])
    recurrent_dropout = eval(sys.argv[5])
    decompose_level = 4 if "stroke" in train_code else 5 if "cji" in train_code else 6
    if vocab == 'pretrain':
        FNAME = f"./results/analogy_0.025/{train_code}.vec"
    else:
        FNAME = f"./vectors/sent_analysis/{train_code}_vectors.txt"
    fin = io.open(FNAME, 'r', encoding='utf-8', newline='\n', errors='ignore')
    
    word_vecs = {}
    for i, line in enumerate(fin):
        tokens = line.rstrip().split(' ')
        array = np.array(list(map(float, tokens[1:])))
        array = array / np.sqrt(np.sum(array * array + 1e-8))
        word_vecs[tokens[0]] = array

    train_data = pd.read_csv(f"./data/parsed_sent_analysis/parsed_sent_analysis_train_{decompose_level}.txt", header=0, delimiter='\t', quoting=3)
    dev_data = pd.read_csv(f"./data/parsed_sent_analysis/parsed_sent_analysis_dev_{decompose_level}.txt", header=0, delimiter='\t', quoting=3)
    test_data = pd.read_csv(f"./data/parsed_sent_analysis/parsed_sent_analysis_test_{decompose_level}.txt", header=0, delimiter='\t', quoting=3)

    text_train = []
    for line in open(f"./data/parsed_sent_analysis/parsed_sent_analysis_train_{decompose_level}.txt", 'r', encoding="utf-8"):
        if line.startswith("id"):
            continue
        words = list(line.split('\t')[1].strip().split())
        text_train.append(words)

    text_dev = []
    for line in open(f"./data/parsed_sent_analysis/parsed_sent_analysis_dev_{decompose_level}.txt", 'r', encoding="utf-8"):
        if line.startswith("id"):
            continue
        words = list(line.split('\t')[1].strip().split())
        text_dev.append(words)

    text_test = []
    for line in open(f"./data/parsed_sent_analysis/parsed_sent_analysis_test_{decompose_level}.txt", 'r', encoding="utf-8"):
        if line.startswith("id"):
            continue
        words = list(line.split('\t')[1].strip().split())
        text_test.append(words)

    if vocab == "train":
        text_to_use = text_train
    else:
        text_to_use = text_train + text_dev + text_test

    tokenizer = Tokenizer(oov_token="<UNK>")
    tokenizer.fit_on_texts(text_to_use)

    MAX_SEQUENCE_LENGTH = 30
    train_sequence = tokenizer.texts_to_sequences(text_train)  # max 47
    train_inputs = pad_sequences(train_sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    train_labels = np.array(train_data['label'])

    dev_sequence = tokenizer.texts_to_sequences(text_dev)  # max 40
    dev_inputs = pad_sequences(dev_sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    dev_labels = np.array(dev_data['label'])

    test_sequence = tokenizer.texts_to_sequences(text_test)  # max 38
    test_inputs = pad_sequences(test_sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    test_labels = np.array(test_data['label'])

    word_size = len(tokenizer.word_index) + 1 
    EMBEDDING_DIM = 300

    embedding_matrix = np.zeros((word_size, EMBEDDING_DIM))

    for word, idx in tokenizer.word_index.items():
        embedding_vector = word_vecs[word] if word in word_vecs else None
        if embedding_vector is not None:
            embedding_matrix[idx] = embedding_vector

    random_seeds = [42, 99]
    acc_total = 0
    precision_total = 0
    recall_total = 0
    f1_total = 0
    for seed in random_seeds:
        tf.keras.backend.clear_session()
        tf.random.set_seed(seed)
        model = Sequential()
        model.add(Embedding(word_size, 300, input_length=MAX_SEQUENCE_LENGTH, weights=[embedding_matrix], trainable=False))
        model.add(LSTM(300, dropout=dropout, recurrent_dropout=recurrent_dropout))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', precision, recall, f1score])

        model.fit(train_inputs, train_labels, epochs=epochs, verbose=0, validation_data=(dev_inputs, dev_labels)) #, callbacks=[es])

        _loss, _acc, _precision, _recall, _f1score = model.evaluate(test_inputs, test_labels, verbose=0)
        acc_total += _acc
        precision_total += _precision
        recall_total += _recall
        f1_total += _f1score
    
    print(f"Results for {train_code}, {vocab}, epochs {epochs}")
    acc_total /= len(random_seeds)
    acc_total *= 100
    precision_total /= len(random_seeds)
    recall_total /= len(random_seeds)
    f1 = 2 / (1/precision_total + 1/recall_total)
    f1_total /= len(random_seeds)
    print('accuracy: {:.2f}%, precision: {:.3f}, recall: {:.3f}, f1score: {:.3f}, f1_average: {:.3f}'.format(acc_total, precision_total, recall_total, f1, f1_total))
    print()


if __name__ == "__main__":
    main()