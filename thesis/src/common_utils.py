import os
import pandas as pd
import glob
import defines
import numpy as np
from  sklearn.metrics import f1_score,recall_score,average_precision_score,precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import json
from  feature_utils import reshape_docs_map_to_seq
def get_doc_idx_from_name(file_name):
    base_name = os.path.basename(file_name)
    return int(base_name.split("_")[0])


def concat_dbs_by_idx(dir_name,db_name,indices,cols=[],index_name=""):
    df_list = []
    for idx in indices:
        df_list.append(os.path.join(os.getcwd(),defines.PATH_TO_DFS,dir_name,"{:02d}_{}.csv".format(idx,db_name)))
    df_list.sort()
    df_map = {}
    for df in df_list:
        df_map[get_doc_idx_from_name(df)] = df
    if len(cols) > 0:
        db = pd.concat([pd.read_csv(i,usecols=cols) for i in df_map.values()],keys=df_map.keys())
    else:
        db = pd.concat([pd.read_csv(i) for i in df_map.values()],keys=df_map.keys())
    db.reset_index(inplace=True)
    new_idx_name = index_name if len(index_name)>0 else "{}_idx".format(db_name.split('_')[0])
    db.rename(columns={'level_0':'doc_idx','level_1':new_idx_name},inplace=True)
    return db

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
    if map_key_is_str(docs_map):
        doc_idx=str(doc_idx)
    seq_idx = np.random.randint(0,len(docs_map[doc_idx]['X']))
    return docs_map[doc_idx]['X'][seq_idx]


def map_key_is_str(docs_map):
    return  isinstance(list(docs_map.keys())[0],str)


def convert_str_keys_to_int(docs_map):
    return {int(k):v for k,v in docs_map.items()}

def get_score(y_true, y_pred, labels,sample_weight=None):
    output_dict=classification_report(
        y_true=y_true,
        y_pred=y_pred,
        labels=labels,
        output_dict=True)
    return output_dict['weighted avg']['f1-score']
    
    
def get_class_weights(y):
    return compute_class_weight(
        class_weight='balanced', classes=np.unique(y), y=y)
    

def get_y_labels(docs_map,indices,seq_len=3,step=3):
    y_l= []
    
    for doc in indices:
        y_l.extend(docs_map[doc]["y_{}_{}".format(seq_len, step)])
    return y_l

def select_dic_keys(docs_map,keys):
    return {key:docs_map[key] for key in keys}



def convert_str_label_to_binary(y):
    return [0 if i == 'not_nar' else 1 for i in y]
    
def convert_binary_label_to_str(y):
    return ['not_nar' if i == 0 else 'is_nar' for i in y]

def get_x_y_by_index(docs_map,indices):
    return select_dic_keys(docs_map,indices),get_y_labels(docs_map,indices)


def get_docs_map(dir_name,docs_map_name,per_par,seq_len,step):
    docs_map=read_docs_map(dir_name,docs_map_name)
    docs_map=convert_str_keys_to_int(docs_map)
    reshape_docs_map_to_seq(docs_map,per_par,seq_len,step)
    add_sent_to_docs_map(dir_name,docs_map)
    return docs_map
    
def read_docs_map(dir_name,docs_map_name="docs_map.json"):
    doc_map_path =  os.path.join(os.getcwd(),defines.PATH_TO_DFS,dir_name,docs_map_name)
    with open(doc_map_path, 'r') as fp:
        docs_map=json.load(fp)
    return docs_map


def add_sent_to_docs_map(dir_name, docs_map):
    for key in docs_map.keys():
        sent_db_path = os.path.join(os.path.join(
            os.getcwd(), defines.PATH_TO_DFS, dir_name, "{:02d}_sent_db.csv".format(int(key))))
        sent_db = pd.read_csv(sent_db_path, usecols=['text', 'is_nar'])
        docs_map[key]['X_bert'] = sent_db['text'].tolist()
        docs_map[key]['y_bert'] = sent_db['is_nar'].tolist()