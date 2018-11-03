import pandas as pd
import random

with open("data/test.txt") as f:
    original = f.read().split("\n")
original = original[:-1]

print(len(original))

def get_mirror(original):
    with open('test_mirror.txt', 'w') as f:
        for sequence in original:
            tokens = sequence.split(" ")
            tokens[5:] = tokens[:5][::-1]
           
            tmp = " ".join(tokens)
            f.write(tmp)
            f.write("\n")
    f.close()

def get_unique(original):
    with open('test_unique.txt', 'w') as f:
        for sequence in original:
            tokens = sequence.split(" ")
            tmp = []
            for i in range(5):
                tmp.append(str(tokens[i]))
                tmp.append(str(tokens[i]))
            tmp = " ".join(tokens)
            f.write(tmp)
            f.write("\n")
    f.close()

def get_shuffle(original):
    with open('test_shuffle.txt', 'w') as f:
        for sequence in original:
            tokens = sequence[:-1].split(" ")
            random.shuffle(tokens)
           
            tmp = " ".join(tokens)
            f.write(tmp)
            f.write("\n")
    f.close()

# get_mirror(original)
# get_unique(original)
get_shuffle(original)
