import os
import sys
from tqdm import tqdm

def get_ngrams(word, minn, maxn):
    word = word.replace("^", "").strip()
    ngrams = set()
    for i in range(len(word)):
        for j in range(minn, maxn+1):
            try:
                ngram = word[i:i+j]
                ngrams.add(ngram)
            except:
                continue
    return ngrams


if __name__ == "__main__":
    with open("data/vector_line/corpus_line_6.txt") as f:
        lines = f.readlines()
    
    dict_12 = set()
    dict_23 = set()
    dict_24 = set()
    dict_39 = set()

    # (2, 3) ~ (3 ~ 12)
    list_hyperparams = [(1, 2), (1, 3)] + [(2, i) for i in range(3, 13)] + [(3, i) for i in range(4, 21)]

    results = {key: set() for key in list_hyperparams}

    for line in tqdm(lines, ncols=100):
        for key in results:
            minn, maxn = key
            results[key].update(get_ngrams(line, minn, maxn))

    print({key: len(results[key]) for key in results})