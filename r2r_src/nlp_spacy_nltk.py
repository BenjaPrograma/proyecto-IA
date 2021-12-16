import json
import string
import spacy
import pickle
from collections import defaultdict
from obj_aware import load_scan_objs_data
import random
import copy
import nltk
#nltk.download('wordnet')
from nltk.corpus import wordnet


import re

import time


def direction_swapper(instr):
    
    for direction in all_directions:
        if direction in instr:
            times_in = instr.count(direction)
            list_of_idx = []
            start = 0
            for _ in range(times_in):
                idx = instr.index(direction,start)
                list_of_idx.append(idx)
                start = idx +1




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


def nltk_instr_objs(save=True):
    basepath = "tasks/R2R/data/"
    lemmatizer = WordNetLemmatizer()
    scan_to_obj = defaultdict(set)
    for split in ["R2R_train.json"]:
        with open(basepath + split, "r") as f:
            new_data = json.load(f)

        all_objs = defaultdict(int)

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
                    if len(wordtype) == 1 and "n" in wordtype:# and not word.endswith("ing"):
                        all_objs[word] +=1
                        scan_to_obj[scan_id].add(word)

        if "train" in split:
            ext = "train_"
        else:
            ext = "val_"

        #print(all_objs)
        sorted_all_objs = dict(sorted(all_objs.items(), key=lambda item: item[1], reverse=True))
        #print(len(sorted_all_objs))
        sorted_all_objs = {k:v for k,v in sorted_all_objs.items() if v > 19}
        #print(len(sorted_all_objs))
        sorted_all_objs.pop("while")
        sorted_all_objs.pop("destination")
        sorted_all_objs.pop("gray")
        sorted_all_objs.pop("threshold")
        sorted_all_objs.pop("path")
        sorted_all_objs.pop("area")
        sorted_all_objs.pop("direction")
        sorted_all_objs.pop("intersection")
        sorted_all_objs.pop("christmas")
        sorted_all_objs.pop("steps")
        sorted_all_objs.pop("step")
        for k,v in sorted_all_objs.items():

            wo = wordnet.synsets(k)
            total = len(wo)
            verb_times = 0
            for w in wo:
                #print(w.name(), end=", ")
                syn = w.name().split('.')
                if syn[1] == "v":
                    verb_times +=1
            
            if verb_times > int(total/2):
                #print(k,v, verb_times, total/2)
                pass
            else:
                print(k,v)#, verb_times, int(total/2), total, wo) #"lemma =",lemmatizer.lemmatize(k))
        if save:
            with open(basepath + "nltk_all_objs_from_instruction_"+ext+ str(len(all_objs))+ ".txt", "w") as f:
                for obj in all_objs:
                    f.write(obj +"\n")

            with open(basepath + 'nltk_all_objs_from_instruction_'+ext+ str(len(all_objs))+ ".pkl", 'wb') as f:
                pickle.dump(all_objs, f, 3)

            with open(basepath + 'nltk_scanid_to_obj_from_instruction'+'.pkl', 'wb') as f:
                pickle.dump(scan_to_obj, f, 3)
        else:
            print("NOT SAVING")


def string_cleaner_nlp(instr):
    instr = instr.lower()
    instr = instr.translate(str.maketrans('','',string.punctuation))
    instr = re.sub(r'\r\n'," ",instr)
    instr = re.sub("\s\s+", " ", instr)
    instr = instr.rstrip().strip()
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


def find_objs_spacy():
    basepath = "tasks/R2R/data/"
    split = "R2R_train.json"
    start = time.time()

    #nlp = spacy.load("en_core_web_sm")
    #spacy.prefer_gpu()
    spacy.require_gpu()

    nlp = spacy.load("en_core_web_trf")
    end = time.time()
    print("TIME TO LOAD SPACY NLP =", str(end-start))
    start = time.time()
    with open(basepath + split, "r") as f:
            new_data = json.load(f)
    ruler = nlp.get_pipe("attribute_ruler")
    patterns = [[{"TEXT":"turn"}],[{"TEXT":"left"}],[{"TEXT":"right"}], 
    [{"TEXT":"walk"}], [{"TEXT":"front"}], [{"TEXT":"end"}],
    [{"TEXT":"top"}],[{"TEXT":"bottom"}],[{"TEXT":"side"}],
    [{"TEXT":"stop"}]
    ]
    attrs = {"POS":"VERB", "POS":"ADV", "POS":"ADV", 
    "POS":"VERB", "POS":"ADV","POS":"ADV", 
    "POS":"ADV", "POS":"ADV", "POS":"ADV", 
    "POS":"VERB"
    }
    ruler.add(patterns=patterns, attrs=attrs) 
    #ruler.add(patterns=patterns, attrs=attrs, index=1) 
    i = 0

    consec_list = []
    for item in new_data:
        scanid = item["scan"]
        for j, instr in enumerate(item["instructions"]):
            instr = string_cleaner_nlp(instr)
            doc = nlp(instr)

            #tagged = pos_tag(word_tokenize(instr))
            #chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}"""
            #chunked = tagstr2tree(tagged)
            #hunked = chunkParser.parse(tagged)
            #chunked.draw()     
            #print(tagged)
            all_consec = []
            consec_nouns = []
            prev_noun = False
            for k,token in enumerate(doc):
                tup = (k,token.text)
                if token.pos_ == "NOUN" and not prev_noun:
                    consec_nouns.append(tup)
                    prev_noun = True
                elif token.pos_ == "NOUN" and prev_noun:
                    consec_nouns.append(tup)
                elif token.pos_ != "NOUN" and prev_noun:
                    all_consec.append(consec_nouns)
                    consec_nouns = []
                    prev_noun = False
                elif token.pos_ != "NOUN" and not prev_noun:
                    pass
            if prev_noun == True:
                all_consec.append(consec_nouns)
            #print(instr)
            #print(all_consec)
            consec_list.append(all_consec)
    

    objects = defaultdict(int)
    for instr_consec in consec_list:
        for consec in instr_consec:
            total_obj = ""
            for tuple in consec:
                total_obj += tuple[1] + " "
            total_obj = total_obj[:len(total_obj)-1]
            objects[total_obj] +=1
    sorted_all_objs = dict(sorted(objects.items(), key=lambda item: item[1], reverse=True))

    threshold = 0 # MIN OCCURRENCE TO BE IN ALL OBJS
    sorted_all_objs = {k:v for k,v in sorted_all_objs.items() if v > threshold}

    basepath = "tasks/R2R/data/"
    with open(basepath + "list_objs_spacy.txt", "w") as f:
        for k,v in sorted_all_objs.items():
            f.write(k + " " +str(v) + "\n")
        
    end = time.time()
    print("TIME ELAPSED =", str(end-start))

def spacy_noun_grouper():
    basepath = "tasks/R2R/data/"
    split = "R2R_train.json"

    with open(basepath + split, "r") as f:
        new_data = json.load(f)

    filename = "list_objs_spacy.txt"
    list_all_objs = {}
    threshold = 10
    with open(basepath + filename, "r") as f:
        for line in f:
            line = line.strip().split(' ')
            cant = int(line.pop())
            text = " ".join(line)
            if cant >= threshold:
                list_all_objs[text] = cant

    basepath = "r2r_src/"
    filename = "not_objs.txt"
    list_not_objs = set()
    with open(basepath + filename, "r") as f:
        for line in f:
            line = line.strip()
            list_not_objs.add(line)

    filename = "directions.txt"
    list_directions_and_contrafactual = dict()
    with open(basepath + filename, "r") as f:
        for line in f:
            line = line.strip().split(',')
            term = line.pop(0)
            list_directions_and_contrafactual[term] = line

    PATHID_TO_OBJ_DIRECTION_IDX = {}
    for item in new_data:
        pathid = item["path_id"]
        PATHID_TO_OBJ_DIRECTION_IDX[pathid] = []

        for j, instr in enumerate(item["instructions"]):
            dir_idx_list = []
            instr = string_cleaner_nlp(instr)
            instr_tok = instr.split(' ')
            max_len_word = 5
            x = 0 
            y = max_len_word + 1
            while x < len(instr_tok):
                found = False
                if y >= len(instr_tok):
                    y = len(instr_tok)
                while x != y:
                    words = instr_tok[x:y]
                    words_str = " ".join(words)
                    if words_str in list_directions_and_contrafactual:
                        dir_idx_list.append((x,y, words_str))
                        x = y
                        y = x + max_len_word + 1
                        found = True
                        break
                    y -=1
                if not found:
                    x +=1
                    y = x + max_len_word + 1

            PATHID_TO_OBJ_DIRECTION_IDX[pathid].append(dir_idx_list)

    basepath = "tasks/R2R/data/"
    with open(basepath + 'pathid_to_direction_idx'+'.pkl', 'wb') as f:
        pickle.dump(PATHID_TO_OBJ_DIRECTION_IDX, f, 3)
    # EXAMPLE:
    #"Walk down one flight of stairs and stop on the landing.",
    #{6250: 

    # 

def load_pathid_to_direction_idx():
    basepath = "tasks/R2R/data/"
    with open(basepath + "pathid_to_direction_idx.pkl", "rb") as f:
        dict = pickle.load(f)
    return dict

def save_opposite_directions_dict():
    basepath = "r2r_src/"
    dicto = dict()
    with open(basepath + "directions.txt", "r") as f:
        for line in f:
            line = line.strip().split(',')
            dir = line.pop(0)
            dicto[dir] = line
    basepath = "tasks/R2R/data/"
    with open(basepath + 'directions_and_contrafactual'+'.pkl', 'wb') as f:
        pickle.dump(dicto, f, 3)

def load_directions_and_contrafactual():
    basepath = "tasks/R2R/data/"
    with open(basepath + "directions_and_contrafactual.pkl", "rb") as f:
        dict = pickle.load(f)
    return dict

def remove_directions(dict_path_direction, instr,i,path_id):
    instr_tok = instr.split(' ')
    idxs = dict_path_direction[path_id][i]
    for tuple in idxs:
        x,y,word = tuple
        while x != y:
            instr_tok[x] = "<UNK>"
            x +=1
    return " ".join(instr_tok)

def contrafactual_directions(dict_path_direction, instr,i,path_id, directions_and_contrafactual):
    instr_tok = instr.split(' ')
    idxs = dict_path_direction[path_id][i]
    idxs_to_pop = []
    for tuple in idxs:
        x,y,word = tuple
        instr_tok[x] = random.choice(directions_and_contrafactual[word])
        if x +1 == y:
            continue
        else:
            for i in range(x+1,y):
                idxs_to_pop.append(i)

    idxs_to_pop.sort(reverse=True)
    for idx in idxs_to_pop:
        instr_tok.pop(idx)
    return " ".join(instr_tok)


def remove_object():
    pass
#def test_nltk_with_instr():
#    nltk_instr_objs(save=False)
#
#test_nltk_with_instr()

#find_objs_nltk()
#find_objs_spacy()


def spacy_obj_instruction():
    basepath = "tasks/R2R/data/"
    split = "R2R_train.json"

    with open(basepath + split, "r") as f:
        new_data = json.load(f)

    filename = "list_objs_spacy.txt"
    list_all_objs = {}
    threshold = 10
    with open(basepath + filename, "r") as f:
        for line in f:
            line = line.strip().split(' ')
            cant = int(line.pop())
            text = " ".join(line)
            if cant >= threshold:
                list_all_objs[text] = cant

    basepath = "r2r_src/"
    filename = "not_objs.txt"
    list_not_objs = set()
    with open(basepath + filename, "r") as f:
        for line in f:
            line = line.strip()
            list_not_objs.add(line)

    filename = "directions.txt"
    list_directions_and_contrafactual = dict()
    with open(basepath + filename, "r") as f:
        for line in f:
            line = line.strip().split(',')
            term = line.pop(0)
            list_directions_and_contrafactual[term] = line

    PATHID_TO_OBJ_IDX = {}
    for item in new_data:
        pathid = item["path_id"]
        PATHID_TO_OBJ_IDX[pathid] = []

        for j, instr in enumerate(item["instructions"]):
            dir_idx_list = []
            instr = string_cleaner_nlp(instr)
            instr_tok = instr.split(' ')
            max_len_word = 6
            x = 0 
            y = max_len_word + 1
            while x < len(instr_tok):
                found = False
                if y >= len(instr_tok):
                    y = len(instr_tok)
                while x != y:
                    words = instr_tok[x:y]
                    words_str = " ".join(words)
                    if words_str in list_all_objs and words_str not in list_not_objs:
                        dir_idx_list.append((x,y, words_str))
                        x = y
                        y = x + max_len_word + 1
                        found = True
                        break
                    y -=1
                if not found:
                    x +=1
                    y = x + max_len_word + 1

            PATHID_TO_OBJ_IDX[pathid].append(dir_idx_list)
    basepath = "tasks/R2R/data/"
    with open(basepath + 'pathid_to_obj_idx'+'.pkl', 'wb') as f:
       pickle.dump(PATHID_TO_OBJ_IDX, f, 3)
    # EXAMPLE:
    #{6250: [[(5, 6, 'stairs')], [(3, 4, 'columns'), (12, 14, 'the steps')], [(7, 8, 'stairs'), (12, 13, 'stairs')]]}

#spacy_obj_instruction()


def load_pathid_to_obj_idx():
    basepath = "tasks/R2R/data/"
    with open(basepath + "pathid_to_obj_idx.pkl", "rb") as f:
        dict = pickle.load(f)
    return dict

def remove_object(pathid_to_obj_idx, instr,i,path_id):
    instr_tok = instr.split(' ')
    idxs = pathid_to_obj_idx[path_id][i]
    for tuple in idxs:
        x,y,word = tuple
        while x != y:
            instr_tok[x] = "<UNK>"
            x +=1
    return " ".join(instr_tok)

def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))


def make_list_of_objs(threshold=10):
    basepath = "tasks/R2R/data/"
    filename = "list_objs_spacy.txt"
    list_all_objs = list()
    with open(basepath + filename, "r") as f:
        for line in f:
            line = line.strip().split(' ')
            cant = int(line.pop())
            text = " ".join(line)
            if cant >= threshold:
                list_all_objs.append(text)
    with open(basepath + 'list_objs_spacy'+'.pkl', 'wb') as f:
       pickle.dump(list_all_objs, f, 3)

def load_list_of_objs():
    basepath = "tasks/R2R/data/"
    with open(basepath + "list_objs_spacy.pkl", "rb") as f:
        dict = pickle.load(f)
    return dict

def replace_object(pathid_to_obj_idx, instr,i,path_id, list_of_objs, alpha=0.5):
    instr_tok = instr.split(' ')
    idxs = pathid_to_obj_idx[path_id][i]
    idxs_to_pop = []
    obj_set = set()
    for tuple in idxs:
        x,y,word =tuple
        obj_set.add(word)
    
    for tuple in idxs:
        x,y,word = tuple
        new_obj = random.choice(list_of_objs)
        #print(word, new_obj)
        while new_obj == "" or new_obj == word or \
            intersection(word.split(' '), new_obj.split(' ')) != [] \
            or new_obj in obj_set:
            new_obj = random.choice(list_of_objs)
        instr_tok[x] = new_obj
        if x +1 == y:
            continue
        else:
            for i in range(x+1,y):
                idxs_to_pop.append(i)

    idxs_to_pop.sort(reverse=True)
    for idx in idxs_to_pop:
        instr_tok.pop(idx)
    return " ".join(instr_tok)



def gen_fake_instruction(pathid_to_direction_idx, pathid_to_obj_idx, directions_and_contrafactual, list_of_objs, instr, j, pathid):
    instr_tok = instr
    instr_objs = pathid_to_obj_idx[pathid][j]
    instr_directions = pathid_to_direction_idx[pathid][j]
    idxs_to_pop = []
    obj_set = set()
    if instr_directions == None:
        what_to_replace = [1]
    else:
        what_to_replace = [1,2,3]
    what_to_replace = random.choice(what_to_replace)
    instr_tok_backup = copy.copy(instr_tok)
    while instr_tok_backup == instr_tok:
        if what_to_replace == 1:
            # SOLO SE REEMPLAZA OBJ

            for tuple in instr_objs:
                _,_,word =tuple
                obj_set.add(word)

            x,y, word = random.choice(instr_objs)
            new_obj = random.choice(list_of_objs)

                # CAMBIAR SOLO 1 OBJ

            while new_obj == word or \
                intersection(word.split(' '), new_obj.split(' ')) != [] \
                or new_obj in obj_set:
                    new_obj = random.choice(list_of_objs)
            instr_tok[x] = new_obj
            if x + 1 == y:
                continue
            else:
                for i in range(x + 1,y):
                    idxs_to_pop.append(i)

        elif what_to_replace == 2:
            x,y, word = random.choice(instr_directions)
            instr_tok[x] = random.choice(directions_and_contrafactual[word])
            if x +1 == y:
                continue
            else:
                for i in range(x+1,y):
                    idxs_to_pop.append(i)


        else:
            # SE REEMPLAZAN 2 COSAS
            for tuple in instr_objs:
                _,_,word =tuple
                obj_set.add(word)

            x,y, word = random.choice(instr_objs)
            new_obj = random.choice(list_of_objs)

                # CAMBIAR SOLO 1 OBJ

            while new_obj == word or \
                intersection(word.split(' '), new_obj.split(' ')) != [] \
                or new_obj in obj_set:
                    new_obj = random.choice(list_of_objs)
            instr_tok[x] = new_obj
            if x + 1 == y:
                continue
            else:
                for i in range(x + 1,y):
                    idxs_to_pop.append(i)
            x,y, word = random.choice(instr_directions)
            ii = 0
            to_ignore = False
            while x+1 in idxs_to_pop:
                x,y, word = random.choice(instr_directions)
                ii +=1
                if ii > 10:
                    to_ignore = True
                    break
            if not to_ignore:
                instr_tok[x] = random.choice(directions_and_contrafactual[word])
                if x +1 == y:
                    continue
                else:
                    for i in range(x+1,y):
                        idxs_to_pop.append(i)
    idxs_to_pop.sort(reverse=True)
    for idx in idxs_to_pop:
        instr_tok.pop(idx)
    return " ".join(instr_tok)

#load_pathid_to_obj_idx()