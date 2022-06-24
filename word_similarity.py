import sys, io
import fasttext
import numpy as np
from tqdm import tqdm
from decompose_letters import jamo_split, reconstruct_word
from scipy import stats

# $ src/fasttext print-word-vectors results/analogy/6_28.bin < data/word_similarity_decomposed/word_similarity_decomposed_stcji.txt > vectors/stcji_vectors.txt


def create_ws_decomposed(level="scji"):
    stroke, cji = False, False
    minjn, maxjn, minn, maxn = 3, 5, 1, 6
    with open("data/WS353_korean.csv", 'r') as f:
        lines = f.readlines()
        with open("word_similarity_decomposed_stcji.txt", 'wt') as g:
            for instance in lines[1:]:
                w1, w2, _ = instance.strip().split(',')
                subw1 = jamo_split(w1, split_stroke=stroke, split_cji=cji)
                subw2 = jamo_split(w2, split_stroke=stroke, split_cji=cji)
                g.write(f"{subw1} {subw2}\n")


def cosine_sim(vec1, vec2):
    vec1 = np.add(vec1, 1e-6 * np.ones_like(vec1))
    vec2 = np.add(vec2, 1e-6 * np.ones_like(vec2))
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def get_vec(line):
    line = line.strip().split(" ")
    word = line[0].strip()
    vec = [eval(x) for x in line[1:]]
    return word, vec





if __name__ == "__main__":
    with open("data/WS353_korean.csv", 'r') as g:
        sims = []
        for line in g.readlines()[1:]:
            word1, word2, sim = line.strip().split(',')
            sims.append(eval(sim))
    
    cosines = []
    MODEL_CODE = sys.argv[1]
    with open(f"vectors/word_similarity/{MODEL_CODE}_vectors.txt", 'r') as f:
        pair = []
        for line in f.readlines():
            word, vec = get_vec(line)
            pair.append(vec)
            if len(pair) == 2:
                cosines.append(cosine_sim(pair[0], pair[1]))
                pair = []
    print(MODEL_CODE)
    print("{:.3f}".format(stats.spearmanr(sims, cosines).correlation))
