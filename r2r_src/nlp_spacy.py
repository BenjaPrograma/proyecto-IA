import json
import string
import spacy
import pickle
from collections import defaultdict

def main():
    spacy.prefer_gpu()
    nlp = spacy.load("en_core_web_trf") # SLOWER BUT MORE ACC
    #nlp = spacy.load("en_core_web_sm") # FASTER BUT LESS ACC

    not_objs = set(["left","right","turn","bottom","top","middle","between"])

    basepath = "tasks/R2R/data/"

    for split in ["R2R_train.json", "R2R_val_seen.json"]:
        with open(basepath + split, "r") as f:
            new_data = json.load(f)

        all_objs = set()

        scan_obj_dict = defaultdict(set)

        for data in new_data:
            scan_id = data["scan"]
            all_text = data["instructions"]
            for text in all_text:
                text = text.lower()
                text = text.strip()
                text = text.translate(str.maketrans('','',string.punctuation))
                #print(text)
                doc = nlp(text)
                prev_was_amod = False
                prev_text = ""
                for token in doc:
                    if token.pos_ in ["PROPN", "NOUN"] and token.text not in not_objs and len(token.text) > 1:
                        all_objs.add(token.text)
                        scan_obj_dict[scan_id].add(token.text)

        if "train" in split:
            ext = "train_"
        else:
            ext = "val_"

        with open(basepath + "spacy_all_objs_from_instruction_"+ext+ str(len(all_objs))+ ".txt", "w") as f:
            for obj in all_objs:
                f.write(obj +"\n")

        with open(basepath + 'spacy_scanid_to_objs_from_instruction_'+ext+'.pkl', 'wb') as f:
            pickle.dump(scan_obj_dict, f, pickle.HIGHEST_PROTOCOL)


def evaluate():
    
    basepath = "tasks/R2R/data/all_objs_file"
    obj_list = []
    with open(basepath + 'spacy_all_objs_from_instruction_val_495.pkl', 'rb') as f:
        obj_list = pickle.load(f)
    
    spacy.prefer_gpu()
    #nlp = spacy.load("en_core_web_trf") # SLOWER BUT MORE ACC
    nlp = spacy.load("en_core_web_sm") # FASTER BUT LESS ACC


#print(scan_obj_dict)    with open(basepath + 'set_objs_certain_218.pkl', 'rb') as f:
        objs_certain = pickle.load(f)