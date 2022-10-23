from collections import Counter
import types
import classes
import feature_utils
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.linear_model import LogisticRegression
from feature_utils import get_prediction_report
from operator import itemgetter
from sklearn.model_selection import LeavePGroupsOut
import time
import common_utils
import numpy as np
from sklearn_crfsuite.utils import flatten
import pandas as pd
import defines
import random
from sklearn_crfsuite import scorers, CRF
from sklearn_crfsuite import metrics
from sklearn_crfsuite.metrics import flat_classification_report
import os
from termcolor import colored, cprint
import json
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import _validation

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted, _num_samples
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from joblib import Parallel, logger
from sklearn.utils.fixes import delayed
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.ensemble._stacking import _BaseStacking
from sklearn.ensemble._base import _fit_single_estimator
from sklearn.base import clone
from sklearn.utils import Bunch
from sklearn.preprocessing._label import LabelEncoder
from sklearn.utils import indexable
import sys
sys.path.append('./src/')

global error_compare_file
global tf_features


def flatten_groups(groups, y):
    return [groups[j] for j, seq in enumerate(y) for i in range(len(seq))]


def get_report_by_unit(cv_db, prefix, unit='split', n_t=2):
    scores = []
    par_scores = []
    full_scores = {}
    par_full_scores = {}
    for split in cv_db['{}_{}'.format(prefix,unit)].unique():
        split_data = cv_db[cv_db['{}_{}'.format(prefix,unit)] == split]
        y_pred = split_data['{}_predicted'.format(prefix)].tolist()
        y_true = split_data['{}_true'.format(prefix)].tolist()
        labels = np.unique(y_true)
        get_prediction_report(y_true, y_pred, np.unique(
            y_true), "{} {}".format(unit,split))
        score, full_score = common_utils.get_report(
            y_true, y_pred, labels, n_t)
        par_y_true, par_y_pred = extract_y_paragraph(
            split_data, prefix, labels)
        if len(par_y_true) == 0:
            raise Exception("par_y_true has 0 length!")
        par_score, par_full_score = common_utils.get_report(
            par_y_true, par_y_pred, labels, n_t)
        scores.append(score)
        par_scores.append(par_score)
        full_scores[split] = full_score
        par_full_scores[split] = par_full_score
    return scores, full_scores, par_scores, par_full_scores


def par_contains_nar(group, kind, prefix, nar_label,not_nar_label):
    return nar_label if nar_label in group['{}_{}'.format(prefix, kind)].unique() else not_nar_label


def extract_y_paragraph(cv_db, prefix, labels):
    db_par = pd.DataFrame()
    doc_col = '{}_group'.format(prefix)
    par_col = '{}_par'.format(prefix)
    true_label = 'is_nar' if isinstance(labels[0], str) else 1
    false_label = 'not_nar' if isinstance(labels[0], str) else 0
    db_par['par_true'] = cv_db.groupby([doc_col, par_col]).apply(
        par_contains_nar, kind='true', prefix=prefix, nar_label=true_label, not_nar_label=false_label)
    db_par['par_predicted'] = cv_db.groupby([doc_col, par_col]).apply(
        par_contains_nar, kind='predicted', prefix=prefix, nar_label=true_label, not_nar_label=false_label)
    return db_par['par_true'].tolist(), db_par['par_predicted'].tolist()


def prepared_cross_validate_ensemble(estimator, cv_db_, prediction_db_, cv_splits, docs_map=None):
    prediction_db = prediction_db_.copy()
    cv_db = cv_db_.copy()
    cols = ['crf_proba_0', 'crf_proba_1', 'bert_proba_0', 'bert_proba_1']

    for split, indices in cv_splits.items():
        single_cv_db = pd.DataFrame()
        print("{} split started...".format(split))
        print("train:", indices['train'])
        print("test:", indices['test'])
        if(estimator.__class__.__name__ == 'CRF'):
            X_train, y_train, X_test, y_test = pack_train_test_for_crf(
                prediction_db, indices, cols, docs_map)
        else:
            X_train, y_train, X_test, y_test = pack_train_test_for_estimator(
                prediction_db, indices, cols)
        y_test_groups = prediction_db[prediction_db['crf_group'].isin(
            indices['test'])]['crf_group']
        ens_clf = estimator.fit(X_train, y_train)
        ens_clf_pred = ens_clf.predict(X_test)
        if(estimator.__class__.__name__ == 'CRF'):
            ens_clf_pred_proba = get_predicted_prob_from_dict(
                flatten(ens_clf.predict_marginals(X_test)))
            ens_clf_pred = flatten(ens_clf_pred)
            y_true = flatten(y_test)
        else:
            ens_clf_pred_proba = ens_clf.predict_proba(X_test)
            y_true = y_test.tolist()
        single_cv_db['ens_predicted'] = ens_clf_pred
        single_cv_db['ens_proba_0'] = ens_clf_pred_proba[:, 0]
        single_cv_db['ens_proba_1'] = ens_clf_pred_proba[:, 1]
        single_cv_db['ens_group'] = y_test_groups.tolist()
        single_cv_db['ens_split'] = int(split)
        single_cv_db['ens_true'] = y_true

        cv_db = pd.concat([cv_db, single_cv_db],
                          ignore_index=True, axis=0, copy=False)
    return cv_db


def pack_train_test_for_estimator(prediction_db, indices, cols):
    X_train = prediction_db[prediction_db['crf_group'].isin(
        indices['train'])][cols]
    y_train = prediction_db[prediction_db['crf_group'].isin(
        indices['train'])]['bert_true']
    X_test = prediction_db[prediction_db['crf_group'].isin(
        indices['test'])][cols]
    y_test = prediction_db[prediction_db['crf_group'].isin(
        indices['test'])]['bert_true']
    return X_train, y_train, X_test, y_test


def pack_train_test_for_crf(prediction_db, indices, cols, docs_map):
    X_flat = []
    for idx, row in prediction_db[prediction_db['crf_group'].isin(indices['train'])].iterrows():
        item = {}
        for feature in cols:
            item[feature] = row[feature]
        X_flat.append(item)
    y = prediction_db[prediction_db['crf_group'].isin(
        indices['train'])]['crf_true']
    X_train = common_utils.reshape_to_seq(X_flat, 8, 8)
    y_train = common_utils.reshape_to_seq(y, 8, 8)
    X_flat = []
    for idx, row in prediction_db[prediction_db['crf_group'].isin(indices['test'])].iterrows():
        item = {}
        for feature in cols:
            item[feature] = row[feature]
        X_flat.append(item)
    y = prediction_db[prediction_db['crf_group'].isin(
        indices['test'])]['crf_true']
    y_test = common_utils.reshape_to_seq(y, 8, 8)
    X_test = common_utils.reshape_to_seq(X_flat, 8, 8)
    return X_train, y_train, X_test, y_test


def prepared_cross_validate_crf(docs_map, cv_splits, seq_len=3, step=3, **crf_params):
    cv_db = pd.DataFrame()
    ftr_db = pd.DataFrame()
    if crf_params:
        print("crf_params passed")
    else:
        print("crf_params not passed")
    for idx, (split, indices) in enumerate(cv_splits.items()):
        single_cv_db = pd.DataFrame()
        print("{} split started for {} train sequences...".format(
            split, len(indices['train'])))
        X_train, y_train, _, _ = get_X_y_by_doc_indices(
            docs_map, indices['train'], seq_len, step)
        X_test, y_test, groups_test, par_test = get_X_y_by_doc_indices(
            docs_map, indices['test'], seq_len, step)
        start_time = time.time()
        if crf_params:
            crf = CRF(
                max_iterations=100,
                all_possible_transitions=True,
                **crf_params)
        else:
            crf = CRF(
                max_iterations=100,
                all_possible_transitions=True,
                algorithm='lbfgs')
        crf.fit(X_train, y_train)
        fit_time = time.time()-start_time
        print("{} split fit of {} samples took {}".format(
            split, len(y_test), time.strftime("%H:%M:%S", time.gmtime(fit_time))))
        single_cv_db['crf_group'] = flatten_groups(groups_test, y_test)
        single_cv_db['crf_par'] = flatten(par_test)
        single_cv_db['crf_split'] = int(split)
        single_cv_db['crf_predicted'] = flatten(crf.predict(X_test))
        predict_time = time.time() - fit_time - start_time
        print("{} split predict took {}".format(
            split, time.strftime("%H:%M:%S", time.gmtime(predict_time))))
        single_cv_db['crf_true'] = flatten(y_test)
        crf_proba = get_predicted_prob_from_dict(
            flatten(crf.predict_marginals(X_test)))
        single_cv_db['crf_proba_0'] = crf_proba[:, 0]
        single_cv_db['crf_proba_1'] = crf_proba[:, 1]
        single_cv_db['crf_sent_idx'] = single_cv_db.index
        # save features from current fold
        if docs_map.__class__.__name__ == 'Dataset':
            single_ftr_db = get_estimator_features(crf, **docs_map.tf_params)
            if idx == 0:
                ftr_db = single_ftr_db
            else:
                ftr_db = ftr_db.merge(single_ftr_db[['label', 'attr', 'weight']], on=['label', 'attr'], suffixes=(
                    "_{}".format(split), None), how='outer', copy=False, validate='one_to_one')

        cv_db = pd.concat([cv_db, single_cv_db],
                          ignore_index=True, axis=0, copy=False)
    return cv_db, ftr_db


def get_info_on_pred(y_pred, y_pred_proba, y_test, groups_test):
    pred_df = pd.DataFrame()
    y_test_flat = flatten(y_test)
    y_pred_flat, pred_df = flatten_sequence(y_pred, groups_test, pred_df)
    y_pred_proba_flat = flatten(y_pred_proba)
    y_pred_proba_max = get_max_predicted_prob(y_pred_proba_flat)
    pred_df["label"] = y_test_flat
    pred_df["pred"] = y_pred_flat
    pred_df["pred_proba"] = y_pred_proba_max
    pred_df["correct"] = np.where(pred_df["label"] == pred_df["pred"], 1, 0)
    err_df = get_sub_pred_db(pred_df, "label != pred")
    corr_df = get_sub_pred_db(pred_df, "label == pred")

    return pred_df, err_df, corr_df


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


def get_predicted_prob_from_dict(y_pred_proba_flat):
    pr_arr = np.zeros((len(y_pred_proba_flat), 2))
    pr_arr[:, 0] = [sample['not_nar'] for sample in y_pred_proba_flat]
    pr_arr[:, 1] = [sample['is_nar'] for sample in y_pred_proba_flat]
    return pr_arr


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


def retrive_predicted_sent(dir_name, pred_df, groups_test, X_test):
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
        sent_db = pd.read_csv(
            os.path.join(
                os.getcwd(),
                defines.PATH_TO_DFS,
                dir_name,
                "{:02d}_sent_db.csv".format(int(doc_idx)),
            )
        )
        for index, row in doc_samples.iterrows():
            err_sent = X_test[int(row["seq_idx"])][int(row["idx_in_seq"])]
            sent_info = sent_db.query(
                "par_idx_in_doc == @err_sent['par_idx_in_doc'] & sent_idx_in_par == @err_sent['sent_idx_in_par']"
            )
            if len(sent_info.index) != 1:
                print("ERROR! Got {} sentenses matching".format(
                    len(sent_info.index)))
            for feature in sent_features:
                pred_df_data.loc[index, feature] = sent_info[feature].values
            pred_df_data.loc[index, "TOKEN"] = err_sent["TOKEN"]
        del sent_db
    return pred_df_data


def get_test_train_idx(docs_map, test_percent, seed=None):
    if not seed is None:
        random.seed(seed)
    doc_indices = set(docs_map.keys())
    doc_count = len(docs_map.keys())
    test_count = int(test_percent * doc_count)
    test_idx = set(random.sample(doc_indices, test_count))
    train_idx = doc_indices - test_idx
    #     print("train {}\ntest {}".format(train_idx,test_idx))
    return train_idx, test_idx


def split_test_train_docs(docs_map, test_percent, seq_len, step, seed=None):
    train_idx, test_idx = get_test_train_idx(docs_map, test_percent, seed)
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    groups_train = []
    groups_test = []
    for idx in train_idx:
        X_train.extend(docs_map[idx]["X_{}_{}".format(seq_len, step)])
        y_train.extend(docs_map[idx]["y_{}_{}".format(seq_len, step)])
        groups_train.extend(
            [idx for i in range(
                len(docs_map[idx]["y_{}_{}".format(seq_len, step)]))]
        )
    for idx in test_idx:
        X_test.extend(docs_map[idx]["X_{}_{}".format(seq_len, step)])
        y_test.extend(docs_map[idx]["y_{}_{}".format(seq_len, step)])
        groups_test.extend(
            [idx for i in range(
                len(docs_map[idx]["y_{}_{}".format(seq_len, step)]))]
        )
    return X_train, y_train, X_test, y_test, test_idx, groups_train, groups_test


def get_X_y_by_doc_indices(docs_map, doc_indices, seq_len, step):
    X = []
    y = []
    groups = []
    par = []
    reshape_name = "{}_{}".format(seq_len, step)
    if isinstance(docs_map, dict):
        X = []
        y = []
        groups = []
        for idx in doc_indices:
            X.extend(docs_map[idx]["X_{}".format(reshape_name)])
            y.extend(docs_map[idx]["y_{}".format(reshape_name)])
            groups.extend(
                [idx for i in range(
                    len(docs_map[idx]["y_{}".format(reshape_name)]))]
            )
    elif isinstance(docs_map, pd.DataFrame):
        X = docs_map[docs_map['doc_idx'].isin(doc_indices)].drop(
            ['doc_idx', 'sent_idx', 'is_nar'], axis=1)
        y = docs_map[docs_map['doc_idx'].isin(doc_indices)]['is_nar']
        groups = docs_map[docs_map['doc_idx'].isin(doc_indices)]['doc_idx']
    else:  # assume it's dataset TBD add type check
        X = docs_map.get_x(doc_indices, reshape_name)
        y = docs_map.get_y(doc_indices, reshape_name)
        groups = docs_map.get_group(doc_indices, reshape_name)
        par = docs_map.get_paragraph(doc_indices, reshape_name)
    return X, y, groups, par


def get_y_by_doc_indices(docs_map, doc_indices, seq_len, step):
    y = []
    for idx in doc_indices:
        y.extend(docs_map[str(idx)]["y_{}_{}".format(seq_len, step)])
    return y


def manual_groups_validate(docs_map, test_percent, seq_len, step, num_splits=10):
    score_list = []
    for i in range(num_splits):
        (
            X_train,
            y_train,
            X_test,
            y_test,
            test_idx,
            groups_train,
            groups_test,
        ) = split_test_train_docs(docs_map, test_percent, seq_len, step)
        score_list.append(manual_get_prediction(
            X_train, y_train, X_test, y_test))
    return np.array(score_list)


def manual_get_prediction(X_train, y_train, X_test, y_test):
    crf = CRF(
        min_freq=5,
        algorithm="lbfgs",
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True,
    )
    crf.fit(X_train, y_train)
    y_pred = crf.predict(X_test)
    labels = list(crf.classes_)
    f1 = metrics.flat_f1_score(
        y_test, y_pred, average="weighted", labels=labels)
    recall = metrics.flat_recall_score(
        y_test, y_pred, average="weighted", labels=labels
    )
    precision = metrics.flat_precision_score(
        y_test, y_pred, average="weighted", labels=labels
    )
    return [f1, recall, precision]


def get_labeles_par_corpus(par_idx, doc_db, print_proba):
    par_corpus = {}
    par_db = doc_db.query("par_idx_in_doc == @par_idx")
    par_corpus["sentenses"] = par_db["text"].tolist()
    par_corpus["pred"] = par_db["pred"].tolist()
    par_corpus["label"] = par_db["label"].tolist()
    par_corpus["is_nar"] = par_db["is_nar"].tolist()
    if print_proba:
        par_corpus["pred_proba"] = par_db["pred_proba"].tolist()
    return par_corpus


def get_labeled_doc_corpus(doc_idx, selected_par_indices, pred_info_df, print_proba):
    doc_corpus = {}
    used_indices = []
    doc_db = pred_info_df.query("doc_idx == @doc_idx")
    for idx in selected_par_indices:
        if idx in used_indices:
            continue
        doc_corpus[idx] = get_labeles_par_corpus(idx, doc_db, print_proba)
        used_indices.append(idx)
        if len(doc_corpus[idx]["sentenses"]) < 2:
            for i in [-1, 1]:
                doc_corpus[idx +
                           i] = get_labeles_par_corpus(idx + i, doc_db, print_proba)
                used_indices.append(idx + 1)
    return doc_corpus


def print_error_par_text(dir_name, indices, pred_info_df, print_proba):
    global error_compare_file
    color_map = {1: "green", 0: "red"}
    nar_args = {1: ["underline"]}
    on_color = {"label": "on_yellow", "pred": "on_cyan"}
    colored_df = pd.DataFrame()
    selected_df = pred_info_df.iloc[indices]
    selected_doc_indices = selected_df["doc_idx"].unique()
    for doc_idx in selected_doc_indices:
        selected_par_indices = selected_df.query("doc_idx == @doc_idx")[
            "par_idx_in_doc"
        ].unique()
        doc_corpus = get_labeled_doc_corpus(
            doc_idx, selected_par_indices, pred_info_df, print_proba)
        # print(
        #     "==========\n{} doc: {} paragraph with error".format(
        #         doc_idx, len(selected_par_indices)
        #     )
        # )
        for par_key in sorted(doc_corpus):
            # print("doc {} par[{}]".format(doc_idx, par_key))
            par_corpus = doc_corpus[par_key]
            # print_labeled_paragraph(par_corpus)
            par_columns = print_labeled_paragraph_by_columns(
                doc_idx, par_key, par_corpus, print_proba
            )
            colored_df = pd.concat(
                [colored_df, par_columns], ignore_index=True)
    colored_df["doc_idx"] = colored_df["doc_idx"].astype(int)
    colored_df["par_idx"] = colored_df["par_idx"].astype(int)
    html = colored_df.to_html(escape=False, justify="center")
    html = r'<link rel="stylesheet" type="text/css" href="df_style.css" /><br>' + html
    # write html to file
    err_report_path = os.path.join(
        os.getcwd(), defines.PATH_TO_DFS, dir_name, "error_analysis.html"
    )
    text_file = open(err_report_path, "w")
    # text_file = open("error_analysis.html", "w")
    text_file.write(html)
    text_file.close()


def get_colored_from_list(_list, true_label=1, html=True):
    text = ""
    corr_style = "<span class='corrStyle'>"
    end = "</span>"
    if html:
        for idx, val in enumerate(_list):
            if val == true_label:
                char = corr_style+"{:03}|".format(idx)+end
            else:
                char = "{:03}|".format(idx)
            text += char
    else:
        for idx, val in enumerate(_list):
            if val == true_label:
                char = colored("{:03}|".format(idx), on_color="on_yellow")
            else:
                char = "{:03}|".format(idx)
            text += char
    return text


def print_labeled_paragraph(par_corpus):

    cprint("Correct labeling:", attrs=["bold"])
    for i, sent in enumerate(par_corpus["sentenses"]):
        cprint(
            text="{}".format(sent),
            on_color=defines.ON_COLOR["label"]
            if par_corpus["label"][i] == "is_nar"
            else None,
            end=".",
        )
    cprint("\nPredicted labeling:", attrs=["bold"])
    for i, sent in enumerate(par_corpus["sentenses"]):
        cprint(
            text="{}".format(sent),
            on_color=defines.ON_COLOR["pred"]
            if par_corpus["pred"][i] == "is_nar"
            else None,
            end=".",
        )
    print("\n")


def print_labeled_paragraph_by_columns(doc_idx, par_idx, par_corpus, print_proba):
    par_columns = pd.DataFrame()
    corr_par = ""
    pred_par = ""
    start = "<span class="
    corr_style = {"is_nar": start + "'corrStyle'>", "not_nar": "<span>"}
    pred_style = {
        "is_nar": start + "'predStyle'>",
        "not_nar": "<span>",
    }

    end = "</span>"
    bold_style = start + "'bold'>"
    for i, sent in enumerate(par_corpus["sentenses"]):
        if print_proba:
            conf_score = (
                bold_style
                + "{:.2f} ".format(par_corpus["pred_proba"][i])
                + end  # + " "
            )
        else:
            conf_score = ""
        corr_par += (
            "".join(
                [conf_score, corr_style[par_corpus["label"][i]], sent, end]) + ". "
        )
        pred_par += (
            "".join(
                [conf_score, pred_style[par_corpus["pred"][i]], sent, end]) + ". "
        )
    par_columns.loc[0, "doc_idx"] = int(doc_idx)
    par_columns.loc[0, "par_idx"] = int(par_idx)
    par_columns.loc[0, "correct"] = corr_par
    par_columns.loc[0, "predicted"] = pred_par

    return par_columns


def print_labeled_paragraph_single_column(doc_idx, par_idx, par_corpus):
    par_columns = pd.DataFrame()
    corr_par = ""
    start = "<span class="
    corr_style = {1: start + "'corrStyle'>", 0: "<span>"}
    end = "</span>"
    for i, sent in enumerate(par_corpus["sentences"]):
        corr_par += (
            "".join(
                [corr_style[par_corpus["label"][i]], sent, end]) + ". "
        )
    par_columns.loc[0, "doc_idx"] = int(doc_idx)
    par_columns.loc[0, "par_idx"] = int(par_idx)
    par_columns.loc[0, "correct"] = corr_par
    par_columns.loc[0, 'par_type'] = par_corpus['par_type']

    return par_columns


def assemble_test_from_parsed(dir_name, doc_idx):
    doc_corpus = pd.read_csv(os.path.join(
        os.getcwd(), defines.PATH_TO_DFS, dir_name, "{:02d}_sent_db.csv".format(doc_idx)))
    colored_df = pd.DataFrame()
    for par_idx in doc_corpus['par_idx_in_doc'].unique():
        par_corpus = {}
        par_corpus['sentences'] = doc_corpus.query(
            'par_idx_in_doc == @par_idx')['text'].tolist()
        par_corpus['label'] = doc_corpus.query(
            'par_idx_in_doc == @par_idx')['is_nar'].tolist()
        par_corpus['par_type'] = doc_corpus.query(
            'par_idx_in_doc == @par_idx')['par_type'].unique()
        par_columns = print_labeled_paragraph_single_column(
            doc_idx, par_idx, par_corpus)
        colored_df = pd.concat(
            [colored_df, par_columns], ignore_index=True)
    colored_df["doc_idx"] = colored_df["doc_idx"].astype(int)
    colored_df["par_idx"] = colored_df["par_idx"].astype(int)
    table = colored_df.to_html(escape=False, justify="center")
    # with open(os.path.join(os.getcwd(),defines.PATH_TO_DFS,"df_style.css"),'r') as f:
    #     style_lines=[]
    #     style_lines.append('<style>')
    #     style_lines.extend(f.readlines())
    #     style_lines.append('<\style>')

    html = colored_df.to_html(escape=False, justify="center")
    html = r'<link rel="stylesheet" type="text/css" href="df_style.css" /><br>' + html
    # write html to file
    err_report_path = os.path.join(
        os.getcwd(), defines.PATH_TO_DFS, dir_name, "{:02d}_assembled.html".format(
            doc_idx)
    )
    text_file = open(err_report_path, "w")
    # text_file = open("error_analysis.html", "w")
    text_file.write(html)
    text_file.close()


def get_test_train_splits(doc_indices, test_doc_num=10, n_splits=3, seed=42):
    cv_splits = {}

    gsf = GroupSplitFold(n_splits=n_splits, n_groups=test_doc_num)
    i = 0
    for tr, ts in gsf.split(X=doc_indices, groups=doc_indices, seed=seed):
        cv_splits[i] = {}
        cv_splits[i]['test'] = np.asarray(itemgetter(*ts)(doc_indices))
        cv_splits[i]['train'] = np.asarray(itemgetter(*tr)(doc_indices))
        i += 1
    return cv_splits


class GroupSplitFold():
    def __init__(self, n_splits=3, n_groups=1, prepared_splits=[]):
        self.n_splits = n_splits
        self.n_groups = n_groups
        self.splits = prepared_splits

    def yeld_prepared_splits(self):
        for split in self.splits:
            yield split.train, split.test

    def split(self, X=None, y=None, groups=None, seed=None):
        if len(self.splits) > 0:
            yield from self.yeld_prepared_splits()
        else:
            doc_indices = set(groups)
            total_test_idx = set(random.sample(
                doc_indices, self.n_groups*self.n_splits))
            if seed:
                random.seed(seed)
            for i in range(self.n_splits):
                test_docs = set(random.sample(total_test_idx, self.n_groups))
                total_test_idx = total_test_idx - test_docs
                train_docs = doc_indices - test_docs
                train_idx = [idx for idx, j in enumerate(
                    groups) if j in train_docs]
                test_idx = [idx for idx, j in enumerate(
                    groups) if j in test_docs]
                yield train_idx, test_idx

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits


class ByDocFold():
    def __init__(self, n_splits=3, n_groups=1, prepared_splits=[]):
        self.n_splits = n_splits
        self.n_groups = n_groups
        self.splits = prepared_splits

    def yeld_prepared_splits(self, X, y, groups):
        for split in self.splits:
            train_idx = [idx for idx, j in enumerate(
                groups) if j in split.train]
            test_idx = [idx for idx, j in enumerate(groups) if j in split.test]
            yield train_idx, test_idx

    def split(self, X, y=None, groups=None):
        if len(self.splits) > 0:
            yield from self.yeld_prepared_splits(X, y, groups)
        else:
            doc_indices = set(groups)

            for i in range(self.n_splits):
                test_docs = set(random.sample(doc_indices, self.n_groups))
                train_docs = doc_indices - test_docs
                train_idx = [idx for idx, j in enumerate(
                    groups) if j in train_docs]
                test_idx = [idx for idx, j in enumerate(
                    groups) if j in test_docs]
                yield train_idx, test_idx

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits


class MyLeavePGroupsOut(LeavePGroupsOut):

    def __init__(self, n_groups, n_splits):
        self.n_groups = n_groups
        self.n_splits = n_splits

    def split(self, X, y=None, groups=None):
        X, y, groups = indexable(X, y, groups)
        indices = np.arange(_num_samples(X))
        iter = 0
        for test_index in super()._iter_test_masks(X, y, groups):
            if iter == self.n_splits:
                break
            train_index = indices[np.logical_not(test_index)]
            test_index = indices[test_index]
            train_index += 1
            test_index += 1
            iter += 1
            yield train_index, test_index

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


class DocsMapFold():
    def __init__(self, n_splits=3):
        self.n_splits = n_splits

    def split(self, X, y=None, groups=None):
        doc_indices = set(X.keys())
        doc_count = len(doc_indices)
        test_count = int(defines.TEST_PERSENT * doc_count)
        for i in range(self.n_splits):
            test_docs = set(random.sample(doc_indices, test_count))
            train_docs = doc_indices - test_docs
            yield train_docs, test_docs

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits


def get_tf_string(attr):
    global tf_features
    string = ""
    if "tfidf" in attr:
        splitted = attr.split("_")
        if "char" in attr:
            tf_type = "char_wb"
        else:
            tf_type = splitted[1]
        tf_idx = int(splitted[-1])
        string = tf_features[tf_type][tf_idx]
    return string


def get_features_df(dir_name, features, tf_name="tf_features_map.json", is_dic=False, **tf_params):
    global tf_features
    tf_features = {}
    json_path = os.path.join(
        os.getcwd(), defines.PATH_TO_DFS, dir_name, tf_name
    )
    if tf_params:
        for k, v in tf_params.items():
            tf_features[k] = v.features
    else:
        with open(json_path, "r") as fp:
            tf_features = json.load(fp)
    features_df = pd.DataFrame()

    if is_dic:
        features_df["weight"] = features.values()
        features_df["label"] = [key[1] for key in list(features.keys())]
        features_df["attr"] = [key[0] for key in list(features.keys())]
    else:
        features_df["weight"] = [key[1] for key in features]
        features_df["label"] = [key[0][1] for key in features]
        features_df["attr"] = [key[0][0] for key in features]
    features_df["string"] = features_df["attr"].transform(get_tf_string)
    del tf_features
    return features_df


class CrfTransformer(TransformerMixin, BaseEstimator):

    def __init__(self, seq_len=3, step=3, param=None):
        print('{}>>>>>>>init() called'.format(self.__class__.__name__))
        self.seq_len = seq_len
        self.step = step
        self.param = param

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X=X).transform(X=X, **fit_params)

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        print('{}>>>>>>>transform() called'.format(self.__class__.__name__))
        X_l = []
        indices = X.keys()
        for doc in indices:
            X_l.extend(X[doc]["X_{}_{}".format(self.seq_len, self.step)])
        return X_l


class CrfClassifier(ClassifierMixin, BaseEstimator):

    def __init__(self, crf_model=None, scorer=None):
        print('{}>>>>>>init() called'.format(self.__class__.__name__))
        self._estimator_type = "classifier"
        self.crf_model = crf_model
        self.labels = ['not_nar', 'is_nar']
        if scorer:
            self.scorer = make_scorer(scorer)
        else:
            self.scorer = make_scorer(metrics.flat_f1_score,
                                      average='weighted', labels=self.labels)
        self.rs_index = -1
        self.rs = {}

    def fit(self, X, y):
        self.classes_ = np.unique(flatten(y))
        self.crf_model.fit(X=X, y=y)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        print('{}>>>>>>>predict() called'.format(self.__class__.__name__))
        check_is_fitted(self, 'is_fitted_')
        return flatten(self.crf_model.predict(X))

    def predict_proba(self, X):
        check_is_fitted(self, 'is_fitted_')
        proba_dict = flatten(self.crf_model.predict_marginals(X))
        return get_predicted_prob_from_dict(proba_dict)

    def score(self, X, y, sample_weight=None):
        print('{}>>>>>>> score() called'.format(self.__class__.__name__))
        return common_utils.get_score(flatten(y), self.predict(X), labels=self.classes_)

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit_transform(self, X, y=None):
        return self.fit(X=X, y=y)

    def set_search_params(self, cv, n_iter, random_state,  **params_space):
        self.params_space = params_space
        self.cv = cv
        self.n_iter = n_iter
        self.random_state = random_state

    def find_best_params(self, X, y, groups, cv=None, n_iter=50, random_state=4, **params_space):
        self.rs_index += 1
        self.set_search_params(cv, n_iter, random_state, **params_space)
        self.rs[self.rs_index] = RandomizedSearchCV(self.crf_model,
                                                    param_distributions=self.params_space,
                                                    cv=self.cv,
                                                    n_iter=self.n_iter,
                                                    n_jobs=-1,
                                                    scoring=self.scorer,
                                                    random_state=self.random_state,
                                                    verbose=2,
                                                    )
        self.rs[self.rs_index].fit(X=X, y=y, groups=groups)
        self.print_rs_result(self.rs_index)

    def print_rs_result(self, iteration):
        print('best params:', self.rs[iteration].best_params_)
        print('best CV score:', self.rs[iteration].best_score_)
        print('model size: {:0.2f}M'.format(
            self.rs[iteration].best_estimator_.size_ / 1000000))

    def predict_on_best_params(self, iteration, X, y):
        if iteration in self.rs.keys():
            y_pred = self.rs[iteration].best_estimator_.predict(X)
            feature_utils.get_prediction_report(flatten(y), flatten(
                y_pred), self.rs[iteration].best_estimator_.classes_)
        else:
            print("ERROR: best parameters for {} was not found yet".format(iteration))


def my_fit_and_score(estimator_pipe, docs_map, scorer, train_idx, test_idx, parameters):
    print("my_fit_and_score called {}".format(time.time()))

    result = {}

    X_train, y_train = common_utils.get_x_y_by_index(docs_map, train_idx)
    X_test, y_test = common_utils.get_x_y_by_index(docs_map, test_idx)

    if parameters is not None:
        # clone after setting parameters in case any parameters
        # are estimators (like pipeline steps)
        # because pipeline doesn't clone steps in fit
        cloned_parameters = {}
        for k, v in parameters.items():
            cloned_parameters[k] = clone(v, safe=False)
            estimator_pipe = estimator_pipe.set_params(**cloned_parameters)
    start_time = time.time()
    estimator_pipe.fit(X_train, y_train)
    fit_time = time.time() - start_time
    test_scores = estimator_pipe.score(X_test, y_test)
    score_time = time.time() - start_time - fit_time
    result["fit_time"] = fit_time
    result["score_time"] = score_time
    result["test_scores"] = test_scores
    print("Current CV score {}".format(test_scores))
    return result


def select_docs_from_map(docs_map, inidces):
    return {key: docs_map[key] for key in inidces}


def select_docs_from_dataset(dataset, inidces):
    return {key: dataset.doc_map[key] for key in inidces}


def my_cross_validate(
    estimator,
    docs_map,
    scoring=None,
    cv=None,
    n_jobs=None,
    verbose=0,
    fit_params=None,
    pre_dispatch="2*n_jobs",
    return_train_score=False,
    return_estimator=False,
    error_score=np.nan,
):

    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose,
                        pre_dispatch=pre_dispatch)
    results = parallel(
        delayed(my_fit_and_score)(
            clone(estimator),
            docs_map,
            scoring,
            train,
            test,
            fit_params
        )
        for train, test in cv.split(docs_map)
    )

    results = _validation._aggregate_score_dicts(results)

    ret = {}
    ret["fit_time"] = results["fit_time"]
    ret["score_time"] = results["score_time"]

    # if return_estimator:
    #     ret["estimator"] = results["estimator"]

    test_scores_dict = _validation._normalize_score_results(
        results["test_scores"])
    # if return_train_score:
    #     train_scores_dict = _validation._normalize_score_results(
    #         results["train_scores"])

    for name in test_scores_dict:
        ret["test_%s" % name] = test_scores_dict[name]
        # if return_train_score:
        #     key = "train_%s" % name
        #     ret[key] = train_scores_dict[name]

    return ret


class MyVotingClassifier(VotingClassifier):

    def __init__(
        self,
        estimators,
        *,
        voting="hard",
        weights=None,
        n_jobs=None,
        flatten_transform=True,
        verbose=False,
    ):
        super().__init__(estimators=estimators)
        self.voting = voting
        self.weights = weights
        self.n_jobs = n_jobs
        self.flatten_transform = flatten_transform
        self.verbose = verbose

    def get_params(self, deep=True):
        return super()._get_params("estimators", deep=deep)

    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.unique(flatten(y)).sort()
        self.le_ = LabelEncoder().fit(flatten(y))
        names, clfs = self._validate_estimators()

        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_single_estimator)(
                clone(clf),
                X,
                y,
                sample_weight=sample_weight,
                message_clsname="Voting",
                message=self._log_message(names[idx], idx + 1, len(clfs)),
            )
            for idx, clf in enumerate(clfs)
            if clf != "drop"
        )

        self.named_estimators_ = Bunch()

        # Uses 'drop' as placeholder for dropped estimators
        est_iter = iter(self.estimators_)
        for name, est in self.estimators:
            current_est = est if est == "drop" else next(est_iter)
            self.named_estimators_[name] = current_est

            if hasattr(current_est, "feature_names_in_"):
                self.feature_names_in_ = current_est.feature_names_in_

        return self

    def score(self, X, y, sample_weight=None):
        return common_utils.get_score(flatten(y), self.predict(X), labels=self.classes_)

    def predict(self, X):
        check_is_fitted(self)
        if self.voting == "soft":
            maj = np.argmax(self.predict_proba(X), axis=1)

        else:  # 'hard' voting
            predictions = self._predict(X)
            maj = np.apply_along_axis(
                lambda x: np.argmax(np.bincount(
                    x, weights=self._weights_not_none)),
                axis=1,
                arr=predictions,
            )
        # maj = common_utils.convert_binary_label_to_str(maj)
        maj = self.le_.inverse_transform(maj)
        return maj


def get_estimator_features(estimator, **tf_params):
    all_features = get_features_df(dir_name="", features=Counter(
        estimator.state_features_).most_common(), tf_name="", is_dic=False, **tf_params)
    return all_features
