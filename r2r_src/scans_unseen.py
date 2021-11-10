import json

import pickle


basepath = "tasks/R2R/data/"
_set = set()
for split in ["R2R_val_unseen.json", "R2R_test.json"]:
    with open(basepath +split, "r") as f:
        new_data = json.load(f)

   
    for data in new_data:
        _set.add(data["scan"])

print(_set)
with open(basepath + 'scans_val_unseen_test_set'+'.pkl', 'wb') as f:
    pickle.dump(_set, f, pickle.HIGHEST_PROTOCOL)


            
#print(scan_obj_dict)