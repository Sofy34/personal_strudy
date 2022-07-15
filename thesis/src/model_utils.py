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
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted,_num_samples
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from joblib import Parallel, logger
from sklearn.utils.fixes import delayed
from sklearn.ensemble import VotingClassifier,StackingClassifier
from sklearn.ensemble._stacking import _BaseStacking
from sklearn.ensemble._base import _fit_single_estimator
from sklearn.base import clone
from sklearn.utils import Bunch
from sklearn.preprocessing._label import LabelEncoder
from sklearn.utils import indexable
import sys
sys.path.append('./src/')
import common_utils
import time
from sklearn.model_selection import LeavePGroupsOut
global error_compare_file
global tf_features


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
    if isinstance(docs_map, dict):
        for idx in doc_indices:
            X.extend(docs_map[idx]["X_{}_{}".format(seq_len, step)])
            y.extend(docs_map[idx]["y_{}_{}".format(seq_len, step)])
            groups.extend(
                [idx for i in range(
                    len(docs_map[idx]["y_{}_{}".format(seq_len, step)]))]
            )
    elif isinstance(docs_map, pd.DataFrame):
        X = docs_map[docs_map['doc_idx'].isin(doc_indices)].drop(
            ['doc_idx', 'sent_idx', 'is_nar'], axis=1)
        y = docs_map[docs_map['doc_idx'].isin(doc_indices)]['is_nar']
        groups = docs_map[docs_map['doc_idx'].isin(doc_indices)]['doc_idx']
    return X, y, groups


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
                doc_corpus[idx + i] = get_labeles_par_corpus(idx + i, doc_db,print_proba)
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


class ByDocFold():
    def __init__(self, n_splits=3, n_groups=1):
        self.n_splits = n_splits
        self.n_groups = n_groups

    def split(self, X, y=None, groups=None):
        doc_indices = set(groups)

        for i in range(self.n_splits):
            test_docs = set(random.sample(doc_indices, self.n_groups))
            train_docs = doc_indices - test_docs
            train_idx = [idx for idx, j in enumerate(groups) if j in train_docs]
            test_idx = [idx for idx, j in enumerate(groups) if j in test_docs]
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
        iter=0
        for test_index in super()._iter_test_masks(X, y, groups):
            if iter == self.n_splits:
                break
            train_index = indices[np.logical_not(test_index)]
            test_index = indices[test_index]
            train_index+=1
            test_index+=1
            iter+=1
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


def get_features_df(dir_name, features, tf_name="tf_features_map.json", is_dic=False):
    global tf_features
    json_path = os.path.join(
        os.getcwd(), defines.PATH_TO_DFS, dir_name, tf_name
    )
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
        indices=X.keys()
        for doc in indices:
            X_l.extend(X[doc]["X_{}_{}".format(self.seq_len, self.step)])
        return X_l


class CrfClassifier(ClassifierMixin, BaseEstimator):

    def __init__(self, crf_model=None):
        print('{}>>>>>>init() called'.format(self.__class__.__name__))
        self._estimator_type = "classifier"
        self.crf_model = crf_model

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
        return self.fit(X=X,y=y)





def my_fit_and_score(estimator_pipe,docs_map,scorer,train_idx,test_idx,parameters):
    print("my_fit_and_score called {}".format(time.time()))

    result = {}
    
    X_train,y_train = common_utils.get_x_y_by_index(docs_map,train_idx)
    X_test,y_test = common_utils.get_x_y_by_index(docs_map,test_idx)

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
                lambda x: np.argmax(np.bincount(x, weights=self._weights_not_none)),
                axis=1,
                arr=predictions,
            )
        # maj = common_utils.convert_binary_label_to_str(maj)
        maj = self.le_.inverse_transform(maj)
        return maj