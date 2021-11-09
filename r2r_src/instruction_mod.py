import json
import numpy as np
    
with open('tasks/R2R/data/R2R_train.json', "r") as f:
    new_data = json.load(f)
#print(new_data[0])

th = 0.8
j = 0

with open('connectivity/scans.txt') as f:
    scans = [scan.strip() for scan in f.readlines()]

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


for img_id, scan in imgid_scan_dict.items():
    long_id = scan + "_" + img_id
    if len(obj_d_feat[long_id]['concat_prob']) > 0:
        for i, feat in enumerate(obj_d_feat[long_id]["concat_feature"]):
            if obj_d_feat[long_id]['concat_prob'][i] < th:
                continue
            obj = obj_d_feat[long_id]["concat_text"][i]
            obj_names_set.add(obj)
            new_imgid_obj_dict[img_id].append(obj)

print(obj_names_set)
new_imgid_obj_dict = {k:list(set(v)) for k,v in new_imgid_obj_dict.items()}
#print(new_imgid_obj_dict)

with open("data/all_objs_dense_0_8_size_"+ str(len(obj_names_set))+ ".txt", "w") as f:
    for obj in obj_names_set:
        f.write(obj +"\n")
