import os
import pandas as pd
import glob
import defines
import numpy as np
def get_doc_idx_from_name(file_name):
    base_name = os.path.basename(file_name)
    return int(base_name.split("_")[0])

def concat_dbs(dir_name,db_name,cols=[]):
    df_list =  glob.glob(os.path.join(os.path.join(os.getcwd(),defines.PATH_TO_DFS,dir_name,"*_{}.csv".format(db_name))))
    df_list.sort()
    df_map = {}
    for df in df_list:
        df_map[get_doc_idx_from_name(df)] = df
    if len(cols) > 0:
        db = pd.concat([pd.read_csv(i,usecols=cols) for i in df_map.values()],keys=df_map.keys())
    else:
        db = pd.concat([pd.read_csv(i) for i in df_map.values()],keys=df_map.keys())
    db.reset_index(inplace=True)
    db.rename(columns={'level_0':'doc_idx','level_1':"{}_idx".format(db_name.split('_')[0])},inplace=True)
    return db

def convert_to_list(_dic):
    dic = {}
    for key,item in _dic.items():
        dic[key] = item.tolist()
    return dic

def convert_to_python_types(_dic):
    dic = {}
    for key,val in _dic.items():
        if not isinstance(val,str):
            dic[key]=val.item()
        else:
            dic[key]=val
    return dic

def get_random_sample(docs_map,seed=None):
    if not seed is None:
        random.seed(seed)
    doc_idx = np.random.randint(1,len(docs_map.keys())+1)
    seq_idx = np.random.randint(0,len(docs_map[doc_idx]['X']))
    return docs_map[doc_idx]['X'][seq_idx]
