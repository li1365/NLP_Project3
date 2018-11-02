import pandas as pd
import random
train = pd.read_csv("data/train.txt", sep = " ", header = None)
print(train.describe())

def get_random():
    res = []
    for i in range(10):
        res.append(str(random.randrange(0,20)))
    res[9] = res[8]
    return res

lists = []
with open('test_unique.txt', 'w') as f:
    for i in range(2000):
        lst = get_random()
        f.write(" ".join(lst))
        f.write('\n')
f.close()