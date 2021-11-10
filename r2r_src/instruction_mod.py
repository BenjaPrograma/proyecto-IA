import json
import numpy as np
import pickle
from collections import defaultdict
basepath = "tasks/R2R/data/"


with open(basepath + 'scans_val_unseen_test_set.pkl', 'rb') as f:
    set_scan_unseen = pickle.load(f)

th = 0.8


with open('connectivity/scans.txt') as f:
    scans = [scan.strip() for scan in f.readlines() if scan.strip() not in set_scan_unseen]

scan_imgid_dict = dict()
for scan in scans:
    scan_imgid_dict[scan] = []
    with open('connectivity/%s_connectivity.json' % scan) as f:
        data = json.load(f)
        for item in data:
            if item["included"]:
                scan_imgid_dict[scan].append(item["image_id"])
        

imgid_scan_dict = {c:k for k,v in scan_imgid_dict.items() for c in v}
new_imgid_obj_dict = {k:[] for k,v in imgid_scan_dict.items()}


filename = "obj_features/0_8/panorama_objs_DenseFeatures_nms1_0_8.npy"
obj_d_feat1 = np.load(filename, allow_pickle=True).item()
filename = "obj_features/0_8/panorama_objs_DenseFeatures_nms2_0_8.npy"
obj_d_feat2 = np.load(filename, allow_pickle=True).item()
obj_d_feat = {**obj_d_feat1, **obj_d_feat2}

#print(obj_d_feat1)
#for i in obj_d_feat1:
#    print(i)
obj_names_set = set()

dense_scanid_obj = defaultdict(set)

for img_id, scan in imgid_scan_dict.items():
    long_id = scan + "_" + img_id
    if len(obj_d_feat[long_id]['concat_prob']) > 0:
        for i, feat in enumerate(obj_d_feat[long_id]["concat_feature"]):
            if obj_d_feat[long_id]['concat_prob'][i] < th:
                continue
            obj = obj_d_feat[long_id]["concat_text"][i]
            obj = obj.strip().split(' ')
            if len(obj) > 1:
                obj.pop(0)
            obj =" ".join(obj)
            obj_names_set.add(obj)
            new_imgid_obj_dict[img_id].append(obj)
            dense_scanid_obj[scan].add(obj)

#print(obj_names_set)
new_imgid_obj_dict = {k:list(set(v)) for k,v in new_imgid_obj_dict.items()}
#print(new_imgid_obj_dict)

with open(basepath + "densefeat_all_objs_" + str(len(obj_names_set))+ ".txt", "w") as f:
    for obj in obj_names_set:
        f.write(obj +"\n")

with open(basepath + "densefeat_all_objs_FOR_EXCL_" + str(len(obj_names_set))+ ".txt", "w") as f:
    for obj in obj_names_set:
        f.write(obj +"\n")

with open(basepath + 'global_scanid_to_imgid'+'.pkl', 'wb') as f:
    pickle.dump(scan_imgid_dict, f, pickle.HIGHEST_PROTOCOL)

with open(basepath + 'global_imgid_to_scanid'+'.pkl', 'wb') as f:
    pickle.dump(imgid_scan_dict, f, pickle.HIGHEST_PROTOCOL)

with open(basepath + 'dense_scanid_to_objs'+'.pkl', 'wb') as f:
    pickle.dump(dense_scanid_obj, f, pickle.HIGHEST_PROTOCOL)
        
with open(basepath + 'dense_imgid_to_objs'+'.pkl', 'wb') as f:
    pickle.dump(new_imgid_obj_dict, f, pickle.HIGHEST_PROTOCOL)
        
