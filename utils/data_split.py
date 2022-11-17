import random

with open('dataset/vialPositioningDataset.txt') as f:
    lines = f.read().splitlines()

list_valid = random.sample(range(len(lines)), int(len(lines)*0.2))

f1 = open("dataset/train.txt", "w")
f2 = open("dataset/valid.txt", "w")

for i in range(len(lines)):
    check = 0
    for j in range(len(list_valid)):
        if i == list_valid[j]:
            f2.write(str(lines[i]) + '\n')
            check = 1

    if check == 0:
        f1.write(str(lines[i]) + '\n')

f1.close()
f2.close()
