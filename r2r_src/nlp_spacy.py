import json
import string
import spacy
import pickle
from collections import defaultdict
from obj_aware import load_scan_objs_data
import random
import copy
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
    
    basepath = "tasks/R2R/data/all_obj_files/"
    obj_list = []
    with open(basepath + 'spacy_all_objs_from_instruction_val_495.txt', 'r') as f:
        for line in f:
            line = line.strip()
            obj_list.append(line)

    spacy.prefer_gpu()
    #nlp = spacy.load("en_core_web_trf") # SLOWER BUT MORE ACC
    nlp = spacy.load("en_core_web_sm") # FASTER BUT LESS ACC

    for obj in obj_list:
        doc = nlp(obj)
        for token in doc:
            if token.pos_ == "VERB":
                print(token.tag_, token.text)

    
#evaluate()


def testing_obj_swapper(alpha=1):

    objs_certain, scanid_to_objs = load_scan_objs_data()
    basepath = "tasks/R2R/data/"
    split = "R2R_train.json"

    with open(basepath + split, "r") as f:
            new_data = json.load(f)

    for item in new_data:
        scanid = item["scan"]
        obj_container = scanid_to_objs[scanid]
            #print(scanid_to_objs)
        instr_objs = set(obj_container[0])
        scan_objs = set(obj_container[1])

        list_objs_certain = list(objs_certain)# FOR EFF

        #print(len(instr_objs), len(scan_objs))
        #print("LEN CERTAIN =", len(objs_certain))
        for j, instr in enumerate(item["instructions"]):
            instr_real = copy.copy(instr)

            instr = instr.lower()
            instr = instr.translate(str.maketrans('','',string.punctuation))
            instr = instr.strip().split(' ')
            i = 0
            while i < len(instr):
                word = instr[i]
                j = 0
                if word in instr_objs:
                    while i+j < len(instr) and instr[i+j] in instr_objs:
                        j +=1
                    j -=1
                    # AGARRAMOS TODOS LOS OBJETOS
                    # QUEDAN DE i HASTA j
                    if alpha >= random.random():
                        # SI ES MAYOR A 1, SE CAMBIA EL OBJ
                        rand_obj_in_scan = True
                        while rand_obj_in_scan:
                            new_obj = random.choice(list_objs_certain)
                            if new_obj in instr_objs or new_obj in scan_objs:
                                continue
                            else:
                                rand_obj_in_scan = False
                        instr[i] = new_obj
                        for _ in range(j):
                            instr.pop(i+1)
                # ELIMINAMOS LOS OBJS CONSECUTIVOS POR 1 DE UNA PALABRA
                i +=1
            instr = " ".join(instr)
            print("real instr =",instr_real)
            print("fake instr =", instr)
        
        


testing_obj_swapper()
    