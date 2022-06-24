import sys, io, os
import fasttext
import numpy as np
from tqdm import tqdm
from decompose_letters import jamo_split, reconstruct_word, reconstruct_word_jamo
from scipy import stats
import pickle
import nltk
nltk.download('punkt')

###### Prepare files ######
def convert_analogy(level=6):
    vocab = set()
    with open(f"./data/parsed_word_analogy/parsed_word_analogy_korean_{level}.txt") as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith(":"):
                continue
            words = line.strip().split()
            for word in words:
                vocab.add(word)
    
    with open(f"./data/vector_line/analogy_words_{level}.txt", 'wt') as g:
        for word in vocab:
            g.write(word + "\n")
    print(len(vocab))

    
def convert_corpus(level=6):
    vocab = dict()
    with open(f"./data/parsed_corpus/parsed_corpus_{level}.txt", 'rt') as corpus:
        for line in tqdm(corpus.readlines(), total=19270697, ncols=100):
            words = line.split()
            for word in words:
                try:
                    vocab[word] += 1
                except KeyError:
                    vocab[word] = 1
    
    print(f"Total number of vocabulary: {len(vocab)}")

    n = 0
    min_count = 10
    with open(f"./data/vector_line/corpus_line_{level}.txt", 'wt') as corpus_line:
        for word, freq in vocab.items():
            if freq >= min_count:
                corpus_line.write(word + "\n")
                n += 1
    
    print(f"Filtered number of vocabulary: {n}")


def convert_sent(level=6):
    vocab = set()
    with open(f"./data_sent_analysis/cleaned_train.txt") as f:
        lines = f.readlines()
        for line in lines:
            line = line.split('\t')[1]
            words = line.strip().split()
            for word in words:
                vocab.add(word)
    
    with open(f"sentiment_words_train.txt", 'wt') as g:
        for word in vocab:
            g.write(word + "\n")
    print(len(vocab))  

if __name__ == "__main__":
    convert_sent()