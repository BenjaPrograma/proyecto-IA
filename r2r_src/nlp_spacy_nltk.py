import json
import string
#import spacy
import pickle
from collections import defaultdict
from obj_aware import load_scan_objs_data
import random
import copy
#import nltk
#nltk.download('wordnet')
from nltk.corpus import wordnet
import re

def main():
    #spacy.prefer_gpu()
    #nlp = spacy.load("en_core_web_trf") # SLOWER BUT MORE ACC
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


def nltk_instr_objs():
    basepath = "tasks/R2R/data/"

    scan_to_obj = defaultdict(set)
    for split in ["R2R_train.json"]:
        with open(basepath + split, "r") as f:
            new_data = json.load(f)

        all_objs = set()

        for data in new_data:
            scan_id = data["scan"]
            all_text = data["instructions"]
            for text in all_text:
                text = text.lower()
                text = text.strip()
                text = text.translate(str.maketrans('','',string.punctuation))
                text = text.split(' ')
                #print(text)
                word_to_type = dict()
                for word in text:
                    wordtype = set()
                    if len(word) > 2:
                        for tmp in wordnet.synsets(word):
                            if tmp.name().split('.')[0] == word:
                                wordtype.add(tmp.pos())
                    word_to_type[word] = wordtype
                    if len(wordtype) == 1 and "n" in wordtype:
                        all_objs.add(word)
                        scan_to_obj[scan_id].add(word)

        if "train" in split:
            ext = "train_"
        else:
            ext = "val_"

        with open(basepath + "nltk_all_objs_from_instruction_"+ext+ str(len(all_objs))+ ".txt", "w") as f:
            for obj in all_objs:
                f.write(obj +"\n")

        with open(basepath + 'nltk_all_objs_from_instruction_'+ext+ str(len(all_objs))+ ".pkl", 'wb') as f:
            pickle.dump(all_objs, f, 3)

        with open(basepath + 'nltk_scanid_to_obj_from_instruction'+'.pkl', 'wb') as f:
            pickle.dump(scan_to_obj, f, 3)

def string_cleaner_nlp(instr):
    instr = instr.lower()
    instr = instr.translate(str.maketrans('','',string.punctuation))
    instr = re.sub(r'\r\n'," ",instr)
    instr = instr.strip()
    return instr

#nltk_instr_objs()
def testing_obj_swapper(alpha=1):

    objs_certain, scanid_to_objs = load_scan_objs_data()
    basepath = "tasks/R2R/data/"
    split = "R2R_train.json"

    real_instr = []
    spacy_fake = []
    nltk_fakes = []
    with open(basepath + split, "r") as f:
            new_data = json.load(f)
    all_objs_list, scan_to_objs = load_nltk_data()
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
            


            instr = instr.split(" ")

            nltk_fake = gen_fake_nltk(all_objs_list, scan_to_objs, copy.copy(instr), scanid)
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
            #print("real instr =",instr_real)
            #print("fake instr =", instr)
            spacy_fake.append(instr)
            #real_instr.append(instr_real)
            nltk_fakes.append(nltk_fake)
            #print(nltk_fake)
    
    with open(basepath + "instr_evaluator.txt", "w") as f:
        for real, fake, fake2 in zip(real_instr,spacy_fake, nltk_fakes):
            f.write("REAL = " + real +"\n")
            #f.write("SPC FAKE = " + fake +"\n")
            f.write("NLTK FAKE = " + fake2 +"\n")

def load_nltk_data():

    basepath = "tasks/R2R/data/nltk_data/"
    all_objs_file = "nltk_all_objs_from_instruction_train_620.pkl"
    scan_objs_file = "nltk_scanid_to_obj_from_instruction.pkl"
    with open(basepath + all_objs_file, "rb") as f:
        all_objs_set = pickle.load(f)

    with open(basepath + scan_objs_file, "rb") as f:
        scan_to_objs = pickle.load(f)
    return list(all_objs_set), scan_to_objs

def gen_fake_nltk(all_objs_list, scan_to_objs, instr, scan_id, alpha=0.5):
    total_only_objs = 0
    word_to_type = dict()
    aux_words = set()
    for word in instr:
        wordtype = set()
        if len(word) > 2:
            for tmp in wordnet.synsets(word):
                if tmp.name().split('.')[0] == word:
                    wordtype.add(tmp.pos())
        word_to_type[word] = wordtype
        if len(wordtype) == 1 and "n" in wordtype:
            total_only_objs +=1
        elif len(wordtype) == 2 and "n" in wordtype:
            aux_words.add(word)

    if len(aux_words) == 0 and total_only_objs == 0:
        # RARE
        return " ".join(instr)
        
    changed_objs = 0
    if total_only_objs != 0:
        while changed_objs == 0:
            i = 0
            while i < len(instr):
                word = instr[i]
                j = 0
                if "n" in word_to_type[word] and len(word_to_type[word]) == 1:
                    while i+j < len(instr) and "n" in word_to_type[instr[i+j]] \
                        and len(word_to_type[instr[i+j]]) == 1:
                        j +=1
                    j -=1
                    if alpha > random.random():
                        rand_obj_in_scan_instr = True
                        while rand_obj_in_scan_instr:
                            new_obj = random.choice(all_objs_list)
                            if new_obj in scan_to_objs[scan_id]:
                                continue
                            else:
                                rand_obj_in_scan_instr = False
                        instr[i] = new_obj
                        changed_objs +=1

                        for _ in range(j):
                            instr.pop(i+1)
                i +=1
            alpha +=alpha/2
    else:
        i = 0

        while changed_objs == 0:
            i = 0
            while i < len(instr):
                word = instr[i]
                j = 0
                if word in aux_words:
                    while i+j < len(instr) and instr[i+j] in aux_words:
                        j +=1
                    j -=1
                    if alpha > random.random():
                        rand_obj_in_scan_instr = True
                        while rand_obj_in_scan_instr:
                            new_obj = random.choice(all_objs_list)
                            if new_obj in scan_to_objs[scan_id]:
                                continue
                            else:
                                rand_obj_in_scan_instr = False
                        instr[i] = new_obj
                        changed_objs +=1

                        for _ in range(j):
                            instr.pop(i+1)
                i +=1
            alpha +=alpha/2

    instr = " ".join(instr)
    return instr

def nltk_remove_obj(instr):
    word_to_type = dict()
    i = 0
    for i in range(len(instr)):
        word = instr[i]
        wordtype = set()
        if len(word) > 2:
            for tmp in wordnet.synsets(word):
                if tmp.name().split('.')[0] == word:
                    wordtype.add(tmp.pos())
        word_to_type[word] = wordtype
        if len(wordtype) == 1 and "n" in wordtype:
            instr[i] = "<UNK>"
    instr = " ".join(instr)
    return instr

def __gen_fake_nltk(instr, scan_objs, instr_objs, list_objs_certain, alpha):
    #instr = instr.strip().split(' ')
    total_only_objs = 0
    word_to_type = dict()
    for word in instr:
        wordtype = set()
        if len(word) > 1:
            for tmp in wordnet.synsets(word):
                if tmp.name().split('.')[0] == word:
                    wordtype.add(tmp.pos())
        word_to_type[word] = wordtype
        if len(wordtype) == 1 and "n" in wordtype:
            total_only_objs +=1
    
    i = 0
    changed_objs = 0
    while i < len(instr):
        word = instr[i]
        j = 0
        if "n" in word_to_type[word] and len(word_to_type[word]) == 1:
            while i+j < len(instr) and "n" in word_to_type[instr[i+j]] \
                and len(word_to_type[instr[i+j]]) == 1:
                j +=1
            j -=1
            if changed_objs < 2 or (changed_objs >= 2 and random.random() > 0.6):
                rand_obj_in_scan = True
                while rand_obj_in_scan:
                    new_obj = random.choice(list_objs_certain)
                    if new_obj in instr_objs or new_obj in scan_objs:
                        continue
                    else:
                        rand_obj_in_scan = False
                instr[i] = new_obj
                changed_objs +=1

                for _ in range(j):
                    instr.pop(i+1)
        i +=1
    instr = " ".join(instr)
    return instr