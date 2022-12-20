import classes
import pickle
import os
import pandas as pd
import glob
import defines
import numpy as np
from sklearn.metrics import f1_score, recall_score, average_precision_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import json
import feature_utils
from itertools import islice
from datetime import datetime


def get_doc_idx_from_name(file_name):
    base_name = os.path.basename(file_name)
    return int(base_name.split("_")[0])


def concat_dbs_by_idx(dir_name, db_name, indices, cols=[], index_name=""):
    df_list = []
    for idx in indices:
        df_list.append(os.path.join(os.getcwd(), defines.PATH_TO_DFS,
                       dir_name, "{:02d}_{}.csv".format(idx, db_name)))
    df_list.sort()
    df_map = {}
    for df in df_list:
        df_map[get_doc_idx_from_name(df)] = df
    if len(cols) > 0:
        db = pd.concat([pd.read_csv(i, usecols=cols)
                       for i in df_map.values()], keys=df_map.keys())
    else:
        db = pd.concat([pd.read_csv(i)
                       for i in df_map.values()], keys=df_map.keys())
    db.reset_index(inplace=True)
    new_idx_name = index_name if len(
        index_name) > 0 else "{}_idx".format(db_name.split('_')[0])
    db.rename(columns={'level_0': 'doc_idx',
              'level_1': new_idx_name}, inplace=True)
    return db


def concat_dbs(dir_name, db_name, cols=[], index_name=""):
    df_list = glob.glob(os.path.join(os.path.join(
        os.getcwd(), defines.PATH_TO_DFS, dir_name, "*_{}.csv".format(db_name))))
    df_list.sort()
    df_map = {}
    for df in df_list:
        df_map[get_doc_idx_from_name(df)] = df
    if len(cols) > 0:
        db = pd.concat([pd.read_csv(i, usecols=cols)
                       for i in df_map.values()], keys=df_map.keys())
    else:
        db = pd.concat([pd.read_csv(i)
                       for i in df_map.values()], keys=df_map.keys())
    db.reset_index(inplace=True)
    new_idx_name = index_name if len(
        index_name) > 0 else "{}_idx".format(db_name.split('_')[0])
    if 'doc_idx' in db.columns:
        doc_idx_str = 'file_idx'
    else:
        doc_idx_str = 'doc_idx'
    db.rename(columns={'level_0': doc_idx_str,
              'level_1': new_idx_name}, inplace=True)
    return db


def convert_to_list(_dic):
    dic = {}
    for key, item in _dic.items():
        dic[key] = item.tolist()
    return dic


def convert_to_python_types(_dic):
    dic = {}
    if isinstance(_dic, dict):
        for key, val in _dic.items():
            if isinstance(val, dict):
                dic[key] = {}
                for subkey, subval in val.items():
                    dic[key][subkey] = convert_item_to_python_types(subval)
            else:
                dic[key] = convert_item_to_python_types(val)
    return dic


def convert_item_to_python_types(val):
    new_val = val
    if not isinstance(val, str):
        if isinstance(val, np.ndarray):
            new_val = val.tolist()
        else:
            new_val = val.item()
    return new_val


def get_random_sample(docs_map, seed=None):
    if not seed is None:
        random.seed(seed)
    doc_idx = np.random.randint(1, len(docs_map.keys())+1)
    if map_key_is_str(docs_map):
        doc_idx = str(doc_idx)
    if 'X' in docs_map[doc_idx]:
        key = 'X'
    else:
        key = 'X_3_3'
    seq_idx = np.random.randint(0, len(docs_map[doc_idx][key]))
    return docs_map[doc_idx][key][seq_idx]


def map_key_is_str(docs_map):
    return isinstance(list(docs_map.keys())[0], str)


def convert_str_keys_to_int(docs_map):
    return {int(k): v for k, v in docs_map.items()}


def get_score(y_true, y_pred, labels, sample_weight=None):
    output_dict = get_report(y_true, y_pred, labels, sample_weight)
    return output_dict['weighted avg']['f1-score']


def get_report(y_true, y_pred, labels, sample_weight=None,n_t=2,segeval=False):
    output_dict = classification_report(
        y_true=y_true,
        y_pred=y_pred,
        labels=labels,
        output_dict=True)
    if segeval:
        my_se=classes.MySegEval(n_t=n_t)
        output_dict['segeval']=my_se.get_scores(y_true,y_pred)
        del output_dict['segeval']['b_stat'] # tbd temporary remove statistic form report to shorten prints
    score = output_dict['weighted avg']['f1-score']
    return score, output_dict


def get_class_weights(y):
    return compute_class_weight(
        class_weight='balanced', classes=np.unique(y), y=y)


def get_y_labels(docs_map, indices, seq_len=3, step=3):
    y_l = []

    for doc in indices:
        y_l.extend(docs_map[doc]["y_{}_{}".format(seq_len, step)])
    return y_l


def get_groups_labels(docs_map, indices, seq_len=3, step=3):
    groups = []
    for doc in indices:
        len_y = len(docs_map[doc]["y_{}_{}".format(seq_len, step)])
        groups.extend([doc for i in range(len_y)])
    return groups


def select_dic_keys(docs_map, keys):
    return {key: docs_map[key] for key in keys}


def convert_str_label_to_binary(y):
    if isinstance(y[0],str):
        return [0 if i == 'not_nar' else 1 for i in y]
    else:
        return y


def convert_binary_label_to_str(y):
    return ['not_nar' if i == 0 else 'is_nar' for i in y]


def get_x_y_by_index(docs_map, indices):
    return select_dic_keys(docs_map, indices), get_y_labels(docs_map, indices)


def get_x_y_group_by_index(docs_map, indices):
    X, y = get_x_y_by_index(docs_map, indices)
    return X, y, get_groups_labels(docs_map, indices)


def get_docs_map(dir_name, docs_map_name, per_par, seq_len, step):
    docs_map = read_docs_map(dir_name, docs_map_name)
    docs_map = convert_str_keys_to_int(docs_map)
    if not 'X_{}_{}'.format(seq_len, step) in docs_map[1]:
        feature_utils.reshape_docs_map_to_seq(docs_map, per_par, seq_len, step)
    add_sent_to_docs_map(dir_name, docs_map)
    return docs_map


def read_docs_map(dir_name, docs_map_name="docs_map.json"):
    doc_map_path = os.path.join(
        os.getcwd(), defines.PATH_TO_DFS, dir_name, docs_map_name)
    with open(doc_map_path, 'r') as fp:
        docs_map = json.load(fp)
    return docs_map


def add_sent_to_docs_map(dir_name, docs_map):
    for key in docs_map.keys():
        sent_db_path = os.path.join(os.path.join(
            os.getcwd(), defines.PATH_TO_DFS, dir_name, "{:02d}_sent_db.csv".format(int(key))))
        sent_db = pd.read_csv(sent_db_path, usecols=['text', 'is_nar'])
        docs_map[key]['X_bert'] = sent_db['text'].tolist()
        docs_map[key]['y_bert'] = sent_db['is_nar'].tolist()


def save_db(db, dir_name, file_name, keep_index=False,  float_format='%.5f'):
    path = os.path.join(os.getcwd(), defines.PATH_TO_DFS,
                        dir_name, "{}.csv".format(file_name))
    print("Saving {}, \nindex {}\nfloat_format {}".format(path, keep_index,float_format))
    db.to_csv(path, index=keep_index,float_format=float_format)


def load_db(dir_name, file_name, keep_index=False):
    path = os.path.join(os.getcwd(), defines.PATH_TO_DFS,
                        dir_name, "{}.csv".format(file_name))
    print("Opened {},  index {}".format(path, keep_index))
    return pd.read_csv(path)


def save_best_params(params, score, dir_name):
    now = datetime.now()
    dt_string = now.strftime("%d.%m_%H:%M")
    score_str = "{:.3f}".format(score).lstrip('0')
    file_name = "{}_{}_best_params.json".format(score_str, dt_string)
    save_json(params, dir_name, file_name)


def save_json(dic_, dir_name, file_name, convert=True):
    if isinstance(dic_, dict):
        dic = dic_.copy()
        if convert:
            dic = convert_to_python_types(dic)
    else:
        dic = dic_
    path = os.path.join(os.getcwd(), defines.PATH_TO_DFS,
                        dir_name, "{}.json".format(file_name))
    print("Saving {}".format(path))
    with open(path, 'w') as fp:
        json.dump(dic, fp)


def write_html(df, path,name):
        html = df.to_html(escape=False, justify="center")
        html = r'<link rel="stylesheet" type="text/css" href="df_style.css" /><br>' + html
        # write html to file
        print_df_path = os.path.join(
            path, "{}.html".format(name)
        )
        text_file = open(print_df_path, "w")
        text_file.write(html)
        text_file.close()


def load_json(dir_name, file_name, convert=True):
    path = os.path.join(os.getcwd(), defines.PATH_TO_DFS,
                        dir_name, "{}.json".format(file_name))
    print("Opened {}".format(path))
    with open(path, 'r') as fp:
        read = json.load(fp)
    return read


def reshape_as_list(lst1, lst2):
    last = 0
    res = []
    for ele in lst1:
        res.append(lst2[last: last + len(ele)])
        last += len(ele)

    return res


def reshape_to_seq(input, seq_len, step):
    return [input[i:i+seq_len] for i in range(0, len(input), step)]


def dump_to_file(_object, dir_name, file_name):
    path = os.path.join(os.getcwd(), defines.PATH_TO_DFS, dir_name,
                        file_name+".p")
    pickle.dump(_object, open(path, "wb"))

def load_pickle(dir_name, file_name):
    path = os.path.join(os.getcwd(), defines.PATH_TO_DFS,dir_name, 
                        file_name+".p")
    return pickle.load(open(path, "rb"))

def get_single_unique(group):
    return group.unique()[0]

def order_meta_features(_meta):
    meta=_meta.copy()
    meta['prefix']=meta['attr'].astype(str).str[:3]
    meta.loc[meta['prefix'].str.contains(r"\-3",na=False),'order']=-3
    meta.loc[meta['prefix'].str.contains(r"\-2",na=False),'order']=1
    meta.loc[meta['prefix'].str.contains(r"\-1",na=False),'order']=2
    meta.loc[~meta['prefix'].str.contains(r"\+|\-",na=False),'order']=3
    meta.loc[meta['prefix'].str.contains(r"\+1",na=False),'order']=4
    meta.loc[meta['prefix'].str.contains(r"\+2",na=False),'order']=5
    meta.loc[meta['prefix'].str.contains(r"\+3",na=False),'order']=6
    meta.sort_values(by=['order','mean'],ascending=False,inplace=True)
    return meta