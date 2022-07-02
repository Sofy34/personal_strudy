import os
import pandas as pd
import glob
import defines

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