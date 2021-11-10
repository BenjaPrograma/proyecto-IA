import pickle
#import numpy as np
import random

def _repickling4():
    basepath = "tasks/R2R/data/useful/"
    with open(basepath + 'set_objs_certain_218.pkl', 'rb') as f:
        objs_certain = pickle.load(f)

    with open(basepath + 'scanid_to_objs_and_aux_objs.pkl', 'rb') as f:
        scanid_to_objs = pickle.load(f)

    with open(basepath + 'set_objs_certain_218.pkl', 'wb') as f:
        pickle.dump(objs_certain, f, 4)

    with open(basepath + 'scanid_to_objs_and_aux_objs.pkl', 'wb') as f:
        pickle.dump(scanid_to_objs, f, 4)

def load_scan_objs_data():
    basepath = "tasks/R2R/data/useful/"
    with open(basepath + 'set_objs_certain_218.pkl', 'rb') as f:

        objs_certain = pickle.load(f)

    with open(basepath + 'scanid_to_objs_and_aux_objs.pkl', 'rb') as f:

        scanid_to_objs = pickle.load(f)
    return objs_certain, scanid_to_objs

#objs_certain, scanid_to_objs = load_scan_objs_data()
#for k,v in scanid_to_objs.items():
#    print(len(v[0]),len(v[1]))
    #print(v[0])
    #print(v[1])
    #break

def swap_objs_using_scanid(objs_certain, scanid_to_objs, scanid, instr, alpha=1):
    obj_container = scanid_to_objs[scanid]
    instr_objs = set(obj_container[0])
    scan_objs = set(obj_container[1])
    print(len(instr_objs), len(scan_objs))
    print("LEN CERTAIN =", len(objs_certain))
    instr = instr.lower().strip().split(" ")
    i = 0
    while i < len(instr):
        word = instr[i]
        j = 0
        while i+j < len(instr) and instr[i+j] in instr_objs:
            j +=1
        j -=1
        # AGARRAMOS TODOS LOS OBJETOS
        # QUEDAN DE i HASTA j
        if alpha >= random.random():
            # SI ES MAYOR A 1, SE CAMBIA EL OBJ
            rand_obj_in_scan = True
            while rand_obj_in_scan:
                new_obj = random.choice(objs_certain)
                if new_obj in instr_objs or new_obj in scan_objs:
                    continue
                else:
                    rand_obj_in_scan = False
            instr[i] = new_obj
            for _ in range(j):
                instr.pop(i+1)
        # ELIMINAMOS LOS OBJS CONSECUTIVOS POR 1 DE UNA PALABRA
    return instr
        


    
    