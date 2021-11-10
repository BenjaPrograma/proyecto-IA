import pickle

basepath = "tasks/R2R/data/"




train_vocab_list = []
with open(basepath + "train_vocab.txt", "r") as f:
    for line in f:
        line = line.strip()
        train_vocab_list.append(line)

with open(basepath + '/useful/set_objs_certain_218.pkl', 'rb') as f:
    objs_certain = pickle.load(f)

for obj in objs_certain:
    if obj not in train_vocab_list:
        train_vocab_list.append(obj)

with open(basepath + "train_vocab.txt", "w") as f:
    for word in train_vocab_list:
        f.write(word+"\n")

with open(basepath + "trainval_vocab.txt", "w") as f:
    for word in train_vocab_list:
        f.write(word+"\n")