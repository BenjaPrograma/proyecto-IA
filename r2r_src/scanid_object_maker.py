import pickle
from collections import defaultdict

basepath = "tasks/R2R/data/all_obj_files/"

with open(basepath + 'set_objs_certain_218.pkl', 'rb') as f:
    objs_certain = pickle.load(f)

basepath = "tasks/R2R/data/"

scanid_to_obj_files = [
    "spacy_scanid_to_objs_from_instruction_train_.pkl",
    "spacy_scanid_to_objs_from_instruction_val_.pkl",
    "matt3d_scanid_to_objs.pkl",
    "dense_scanid_to_objs.pkl"
    ]
# JUST TO ITER OVER SCANID's
with open(basepath + 'global_scanid_to_imgid.pkl', 'rb') as f:
    scan_id_iter = pickle.load(f)


with open(basepath + scanid_to_obj_files[0], "rb") as f:
    spacy_train = pickle.load(f)
with open(basepath + scanid_to_obj_files[1], "rb") as f:
    spacy_val = pickle.load(f)
with open(basepath + scanid_to_obj_files[2], "rb") as f:
    matt3d = pickle.load(f)
with open(basepath + scanid_to_obj_files[3], "rb") as f:
    dense = pickle.load(f)

scanid_to_objs = defaultdict(list)

for scan_id, _ in scan_id_iter.items():

    
    spacy_train_obj = spacy_train[scan_id]
    spacy_val_obj = spacy_val[scan_id]
    spacy = list(spacy_val_obj.union(spacy_train_obj))
    matt3d_obj = matt3d[scan_id]
    dense_obj = dense[scan_id]
    scanid_to_objs[scan_id].append(spacy)

    obs_obj = set()
    for obj in matt3d_obj:
        if obj in objs_certain:
            obs_obj.add(obj)
    for obj in dense_obj:
        if obj in objs_certain:
            obs_obj.add(obj)
    obs_obj = list(obs_obj)
    scanid_to_objs[scan_id].append(obs_obj)

with open(basepath + 'scanid_to_objs_and_aux_objs' + '.pkl', 'wb') as f:
    pickle.dump(scanid_to_objs, f, pickle.HIGHEST_PROTOCOL)    
