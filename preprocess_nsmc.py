import re
import random

def clean_line(line):
    line = re.sub(r'[^\w\s]|[ㄱ-ㅎ]|[ㅏ-ㅣ]', ' ', line)
    return line.strip()


def preprocess_data():
    with open("./data/sent_analysis/nsmc.txt", 'rt') as f:
        lines = f.readlines()

    with open("./data/sent_analysis/nsmc_pos.txt", 'wt') as g_pos:
        with open("./data/sent_analysis/nsmc_neg.txt", 'wt') as g_neg:
            for line in lines[1:]:
                try:
                    id_num, doc, label = line.strip().split('\t')
                except:
                    print(line)
                    continue
                doc = clean_line(doc)
                doc = " ".join(doc.split())
                if label == "1":
                    g_pos.write(f"{id_num}\t{doc}\t{label}\n")
                else:
                    g_neg.write(f"{id_num}\t{doc}\t{label}\n")

def split_data():
    pos = open("./data/sent_analysis/nsmc_pos.txt", 'rt')
    pos_list = [line.strip() for line in pos.readlines()]
    pos.close()
    neg = open("./data/sent_analysis/nsmc_neg.txt", 'rt')
    neg_list = [line.strip() for line in neg.readlines()]
    neg.close()

    random.shuffle(pos_list)
    random.shuffle(neg_list)

    train = open("./data/sent_analysis/cleaned_train.txt", 'wt')
    dev = open("./data/sent_analysis/cleaned_dev.txt", 'wt')
    test = open("./data/sent_analysis/cleaned_test.txt", 'wt')

    pos_train, pos_dev, pos_test = pos_list[:50000], pos_list[50000:62500], pos_list[62500:75000]
    neg_train, neg_dev, neg_test = neg_list[:50000], neg_list[50000:62500], neg_list[62500:75000]

    train.write("id\tdocument\tlabel\n")
    dev.write("id\tdocument\tlabel\n")
    test.write("id\tdocument\tlabel\n")

    for p, n in zip(pos_train, neg_train):
        train.write(p + "\n")
        train.write(n + "\n")
    
    for p, n in zip(pos_dev, neg_dev):
        dev.write(p + "\n")
        dev.write(n + "\n")
    
    for p, n in zip(pos_test, neg_test):
        test.write(p + "\n")
        test.write(n + "\n")
    
    train.close()
    dev.close()
    test.close()

if __name__ == "__main__":
    random.seed(42)
    preprocess_data()
    split_data()