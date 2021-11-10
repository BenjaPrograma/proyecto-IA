import pickle

basepath = "tasks/R2R/data/all_obj_files/"

all_objs_files = ["matt3d_all_objs_792", 
    "spacy_all_objs_from_instruction_train_1780",
    "spacy_all_objs_from_instruction_val_495",
    "densefeat_all_objs_FOR_EXCL_520"
]

all_objs_sets = []

for file in all_objs_files:
    new_set = set()
    with open(basepath + file + ".txt", "r") as f:
        for line in f:
            line = line.strip()
            if len(line) > 1:
                new_set.add(line)
    all_objs_sets.append(new_set)


spacy_set = all_objs_sets[1].union(all_objs_sets[2])
final_set = all_objs_sets[0] &  spacy_set & all_objs_sets[3]

with open(basepath + "objs_certain_" + str(len(final_set))+ ".txt", "w") as f:
    for obj in final_set:
        f.write(obj +"\n")

with open(basepath + 'set_objs_certain_'+ str(len(final_set)) + '.pkl', 'wb') as f:
    pickle.dump(final_set, f, pickle.HIGHEST_PROTOCOL)