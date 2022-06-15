import numpy as np
from sklearn_crfsuite.utils import flatten
import pandas as pd
import defines


def error_analisys(y_pred, y_pred_proba, y_true, groups_test):
    err_df = get_err_df(y_pred, y_pred_proba, y_true, groups_test)
    return err_df


def get_err_df(y_pred, y_pred_proba, y_test, groups_test):
    pred_df = pd.DataFrame()
    y_test_flat = flatten(y_test)
    y_pred_flat, pred_df = flatten_sequence(y_pred, groups_test, pred_df)
    y_pred_proba_flat = flatten(y_pred_proba)
    y_pred_proba_max = get_max_predicted_prob(y_pred_proba_flat)
    pred_df["label"] = y_test_flat
    pred_df["pred"] = y_pred_flat
    pred_df["pred_proba"] = y_pred_proba_max

    err_df = get_sub_pred_db(pred_df, "label != pred")
    corr_df = get_sub_pred_db(pred_df, "label == pred")

    return err_df, corr_df


def get_sub_pred_db(pred_df, query):
    sub_df = pred_df.query(query)
    sub_df = sub_df.sort_values(by="pred_proba", ascending=False)
    sub_df.reset_index(drop=True, inplace=True)
    return sub_df


def flatten_sequence(samples, groups_test, pred_df):
    for i, sample in enumerate(samples):
        for j, seq_item in enumerate(sample):
            idx = pred_df.shape[0]
            pred_df.loc[idx, "seq_idx"] = i
            pred_df.loc[idx, "idx_in_seq"] = j
            pred_df.loc[idx, "doc_idx"] = groups_test[i]
            pred_df.loc[idx, "seq_len"] = len(sample)
    y_pred_flat = flatten(samples)
    return y_pred_flat, pred_df


def get_max_predicted_prob(y_pred_proba_flat):
    return [max(sample.values()) for sample in (y_pred_proba_flat)]


def get_sample_info(X_test, _pred_df):
    """[summary]

    Args:
        X_test ([list of lists]): [Test samples: sequences of sentences]
        _pred_df ([DataFrame]): [predicted and actual labels]

    Returns:
        [DataFrame]: [extended dataframe with sample features]
    """
    pred_df = _pred_df.copy()
    for idx, row in _pred_df.iterrows():
        err_sent = X_test[int(row["seq_idx"])][int(row["idx_in_seq"])]
        for key in defines.SAMPLE_FEATURES:
            pred_df.loc[idx, key] = err_sent[key]
    return pred_df


def retrive_predicted_sent(pred_df, groups_test, X_test):
    # """[summary]

    # Parameters
    # ----------
    # pred_df : [DataFrame]
    #     [predicted and tru label per sample with confidence score]
    # groups_test : [list of lists]
    #     [doc index for each item in sequences]
    # X_test : [list of list]
    #     [samples features]

    # Returns
    # -------
    # [DataFrame]
    #     [predicted df with addition of sentense info]
    # """
    sent_features = defines.SENT_FEATURES
    sent_features.extend(["text"])
    pred_df_data = pred_df.copy()
    test_docs = pred_df_data["doc_idx"].unique()
    for doc_idx in test_docs:
        doc_samples = pred_df.query("doc_idx == @doc_idx")
        sent_db = pd.read_csv(r"./dataframes/{:02d}_sent_db.csv".format(int(doc_idx)))
        for index, row in doc_samples.iterrows():
            err_sent = X_test[int(row["seq_idx"])][int(row["idx_in_seq"])]
            sent_info = sent_db.query(
                "par_idx_in_doc == @err_sent['par_idx_in_doc'] & sent_idx_in_par == @err_sent['sent_idx_in_par']"
            )
            if len(sent_info.index) != 1:
                print("ERROR! Got {} sentenses matching".format(len(sent_info.index)))
            for feature in sent_features:
                pred_df_data.loc[index, feature] = sent_info[feature].values
            pred_df_data.loc[index, "TOKEN"] = err_sent["TOKEN"]
        del sent_db
    return pred_df_data

    
