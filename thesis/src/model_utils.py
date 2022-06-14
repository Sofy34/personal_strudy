import numpy as np
from sklearn_crfsuite.utils import flatten
import pandas as pd

def error_analisys(y_pred,y_pred_proba,y_true,groups_test):
    err_df = get_err_df(y_pred,y_pred_proba,y_true,groups_test)
    return err_df

def get_err_df(y_pred,y_pred_proba,y_test,groups_test):
    pred_df = pd.DataFrame()
    y_test_flat = flatten(y_test)
    y_pred_flat,pred_df = flatten_sequence(y_pred,groups_test,pred_df)
    y_pred_proba_flat = flatten(y_pred_proba)
    y_pred_proba_max = get_max_predicted_prob(y_pred_proba_flat) 
    pred_df['label']=y_test_flat
    pred_df['pred'] = y_pred_flat
    pred_df['pred_proba']=y_pred_proba_max
    
    err_df = get_sub_pred_db(pred_df,"label != pred")
    corr_df = get_sub_pred_db(pred_df,"label == pred")

    return err_df,corr_df

def get_sub_pred_db(pred_df,query):
    sub_df = pred_df.query(query)
    sub_df = sub_df.sort_values(by='pred_proba',ascending=False)
    sub_df.reset_index(drop=True,inplace=True)
    return sub_df

def flatten_sequence(samples,groups_test,pred_df):
    for i,sample in enumerate(samples):
        for j,seq_item in enumerate(sample):
            idx = pred_df.shape[0]
            pred_df.loc[idx,'seq_idx'] = i
            pred_df.loc[idx,'idx_in_seq'] = j
            pred_df.loc[idx,'doc_idx'] = groups_test[i]
            pred_df.loc[idx,'seq_len'] = len(sample)
    y_pred_flat = flatten(samples)
    return y_pred_flat,pred_df

        

def get_max_predicted_prob(y_pred_proba_flat):
    return [max(sample.values()) for sample in (y_pred_proba_flat)]



