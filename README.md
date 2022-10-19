# Break it Down Into BTS : Basic, Tiniest Subword Units for Korean

_Nayeon Kim*, Jun-Hyung Park*, Joon-Young Choi, Eojin Jeon, Youjin Kang, and SangKeun Lee (* equal contribution)_

_Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing (EMNLP 2022)_

This repository contains the code for our EMNLP 2022 paper.

## Training
### 0. Build
Before you start training Korean word vectors, you should build the source by using `make`.
```
$ cd src
$ make
```
This will produce object files for all the classes as well as the main binary `fasttext`.


### 1. Parse Korean documents.
First, You should parse a Korean document with `decompose_letters.py`. An example use is as follows:

```
python decompose_letters.py [input_file_name] [parsed_file_name] [split_stroke] [split_cji]
```
Note that split_stroke and split_cji are boolean values to be set as True or False.

SISG(stroke) requires corpus parsed with split_stroke=True, split_cji=False.

SISG(cji) requires corpus parsed with split_stroke=False, split_cji=True.

SISG(BTS) requires corpus parsed with split_stroke=True, split_cji=True.

We decompose the letters into cji/stroke levels. If you want to compose them into character/word level, you can modify the parsing method by adding a special seperator. Refer to the following example.

For example, if we decompose the word "이라크" with above settings respectively with character-level seperator <b>/</b> (chosen to be a character that does not appear in the corpus)

+ SISG(stroke): ㅇㅣ/ㄹㅏ/ㄱ-ㅡ/
+ SISG(cji): ㅇㅣ/ㄹㅣㆍ/ㅋㅡ/
+ SISG(BTS): ㅇㅣ/ㄹㅣㆍ/ㄱ-ㅡ/


### 2. Train Korean word vectors.
Then, you can train word vectors for Korean by executing the complied source. The source code will accept the output file `[parsed_file_name]`  generated by `decompose_letters.py`. An example use case is as follows:

```
[fastText_executable_path] skipgram -input [parsed_file_name] -output [output_file_name] -minCount 10 -minsc 3 -maxsc [6-20] -minjn 1 -maxjn 0 -minn 1 -maxn 0 -dim 300 -ws 5 -epoch 5 -neg 5 -loss ns -thread 16 -lr 0.025
```

The full list of parameters are given below.

```
-minCount : minimal number of word occurences [5]
-bucket : number of buckets [10000000]
-minsc: min length of stroke/cji ngram
-maxsc: max length of stroke/cji ngram
-minn : min length of char ngram [1]
-maxn : max length of char ngram [4]
-minjn : min length of jamo ngram [3]
-maxjn : max length of jamo ngram [5]
-t : sampling threshold [1e-4]
-lr : learning rate [0.05]
-dim : size of word vectors [100]
-ws : size of the context window [5]
-loss : loss function {ns, hs, softmax} [softmax]
-neg : number of negatives sampled [5]
-epoch : number of epochs [5]
-thread : number of threads [12]
-verbose : verbosity level [2]
```

The default number of character-level n-grams is set to 1-6, and the number of jamo-level n-grams is set to 3-6. I have not identified any problem from the bucket size, you can use the default bucket size (10,000,000) in our experiments.


### 3. Evaluate on analogies.
You can evaluate the trained word vectors on analogies by typing as follows: 

```
[fastText_executable_path] analogies [output_file_name] [parsed_analogy_data_file_name]
```

[output_file_name] corresponds to the .bin file created from training.

You must prepare the corresponding parsed analogy data file for SISG(stroke), SISG(cji), and SISG(BTS), so that the parsing methods of the corpus and word analogy file are aligned. 

```
python decompose_letters.py [word_analogy_file_name] [parsed_analogy_data_file_name] [split_stroke] [split_cji]
```


### Constructing Korean OOV word vectors
The trained output file `[output_file_name].bin` can be used to compute word vectors for OOVs. Provided you have a text file `queries.txt` containing Korean decomposed words for which you want to compute vectors, use the following command:

```
$ [fastText_executable_path] print-word-vectors model.bin < queries.txt
```

Note that  `queries.txt` should contain decomposed Korean words, such as ㄱㅣㆍㅇ/ㅇㅣㆍ/ㅈㅣ/ for 강아지. You can also use `jamo_split` method in `decompose_letters.py` to obtain decomposed Korean words.


## Change Log
19-10-22 : Initial upload. version 1.0
