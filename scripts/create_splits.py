#requirement: pip install conllu

import os
from conllu import parse as conllu_parse
import random


def replicability_function():
  return 0.2


def create_splits(path):
    os.chdir(path)

    print("Splitting...")

    with open("train.conllu", encoding='utf-8') as reader:
        sentences = conllu_parse(reader.read())

    random.shuffle(sentences, replicability_function)

    print(f"Total number of sentences: {len(sentences)}")
    train_sentences = sentences[:len(sentences)//100*90]
    dev_sentences = sentences[len(sentences)//100*90:len(sentences)//100*95]
    test_sentences = sentences[len(sentences)//100*95:]

    print(f"Total number of train sentences: {len(train_sentences)}")
    print(f"Total number of dev sentences: {len(dev_sentences)}")
    print(f"Total number of test sentences: {len(test_sentences)}")


    out = open("train.conllu", "w")
    for sentence in train_sentences:
        for i in range(len(sentence)):
            out.write(str(i)+"\t"+sentence[i]["form"]+"\t"+sentence[i]["lemma"]+"\n")
        out.write("\n")

    out = open("dev.conllu", "w")
    for sentence in dev_sentences:
        for i in range(len(sentence)):
            out.write(str(i)+"\t"+sentence[i]["form"]+"\t"+sentence[i]["lemma"]+"\n")
        out.write("\n")

    out = open("test.conllu", "w")
    for sentence in test_sentences:
        for i in range(len(sentence)):
            out.write(str(i)+"\t"+sentence[i]["form"]+"\t"+sentence[i]["lemma"]+"\n")
        out.write("\n")

    return



path = input("Insert the relative path to folder in which there is the file to split (e.g., ../data/wikineural/en/): ")
create_splits(path)