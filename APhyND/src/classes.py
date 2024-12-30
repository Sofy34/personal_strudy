import seaborn as sns

from sklearn.metrics import roc_curve, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix
from operator import itemgetter
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, recall_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import copy
import time
import segeval as se
import model_utils
from sklearn_crfsuite.utils import flatten
from sklearn import metrics
import feature_utils
import common_utils
import sys
import pandas as pd
import os
from scipy import sparse
import defines
import numpy as np
import pickle
import re

from sklearn.metrics import make_scorer, accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import BaseEstimator, TransformerMixin, MetaEstimatorMixin, ClassifierMixin
sys.path.append('./src/')


class TfParams:

    def __init__(self, dir_name, tf_type,  split_idx, splits, stop_list=[]):
        self.tf_type = tf_type
        self.stop_list = stop_list
        self.split_idx = split_idx
        self.splits = splits
        self.dir_name = dir_name
        self.suffix = '_no.stop' if len(
            stop_list) == 0 else '_stop{}'.format(len(stop_list))
        self.tf = None
        self.features = None
        self.set_tf_params()
        self.fit_train(self.splits[self.split_idx]['train'])
        self.transform_save(self.splits[self.split_idx]['train'])
        self.transform_save(self.splits[self.split_idx]['test'])
        print('>> init {} for split {}: total {} features'.format(
            self.tf_type, self.split_idx, len(self.features)))

    def set_tf_params(self):
        if self.tf_type == 'char_wb':
            self.per_word = False
            self.per_lemma = False
            self.analyzer = 'char_wb'
        elif self.tf_type == 'lemma':
            self.per_word = True
            self.per_lemma = True
            self.analyzer = 'word'
        elif self.tf_type == 'word':
            self.per_word = True
            self.per_lemma = False
            self.analyzer = 'word'
        else:
            print("ERROR: unknown TfIdf type {}".format(self.tf_type))

    def fit_train(self, doc_indices):
        self.tf = feature_utils.tfidf_selected_fit(
            dir_name=self.dir_name,
            per_word=self.per_word,
            per_lemma=self.per_lemma,
            analyzer=self.analyzer,
            doc_indices=doc_indices)
        self.features = self.tf.get_feature_names_out()

    def transform_save(self, doc_indices):
        for doc in doc_indices:
            doc_tf = feature_utils.tfidf_transform_doc(
                self.dir_name, doc, self.tf, self.per_lemma)
            common_utils.save_sparse(self.dir_name, "{:02d}_{}_tfidf_{}{}.npz".format(
                doc, self.split_idx, self.tf_type, self.suffix), doc_tf)


class Sentence:
    def __init__(self,
                 doc_idx,
                 doc_len,
                 par_idx,
                 sent_idx,
                 text,
                 par_type,
                 nar_idx):
        self.doc_idx = doc_idx
        self.sent_idx = sent_idx
        self.doc_len = doc_len
        self.par_idx = par_idx
        self.x = {}
        self.y = None
        self.pred_y = None
        self.text = text
        self.par_type = par_type
        self.nar_idx = nar_idx
        self.corr_style = {
            "is_nar": "<span class='corrStyle'>", "not_nar": "<span>"}
        self.end = "</span>. "

    def set_features(self, feature_dict, split_idx):
        if type(feature_dict) != dict:
            print("ERROR: not a dict")
        else:
            self.x[split_idx] = feature_dict

    def set_y(self, y=None):
        self.y = y

    def get_x(self, split_idx):
        return self.x[split_idx]

    def get_text(self):
        return self.text

    def get_y(self):
        return self.y

    def get_paragraph(self):
        return self.par_idx

    def set_pred_y(self, pred_y):
        self.pred_y = pred_y

    def get_pred_y(self):
        return self.pred_y

    def print(self, label="true"):
        if label == "pred" and self.pred_y == None:
            raise Exception("Can't print predicted since label is not given")
        return "".join(
            [self.corr_style[self.y if label == "true" else self.pred_y], self.text, self.end])


class Paragraph():
    def __init__(self,
                 doc_idx,
                 par_idx,
                 par_type):
        self.doc_idx = doc_idx
        self.par_idx = par_idx
        self.sent_list = []
        self.y = 'not_nar'
        self.par_len = 0
        self.nar_sent_count = 0
        self.par_type = par_type

    def set_y(self, y):
        self.y = y

    def add_sent(self, sent):
        self.sent_list.append(sent)
        if sent.get_y() == 'is_nar':
            self.nar_sent_count += 1
            # if at least one sentence is narrative => paragraph is narrative
            self.set_y('is_nar')
        self.par_len += 1

    def print(self, label="true"):
        text = ""
        for sent in self.sent_list:
            text += sent.print(label)
        return text

    def get_par_type(self):
        return self.par_type


class Document:
    def __init__(self,
                 idx,
                 path,
                 tf_params={},
                 splits={},
                 merged_str="merged_db",
                 neighbor_radius=3):
        self.sent_list = []
        self.splits = splits
        self.reshaped = {}
        self.doc_idx = idx
        self.path = path
        self.tf_params = tf_params
        self.merged_str = merged_str
        self.neighbor_radius = neighbor_radius
        self.doc_db = {}
        self.doc_len = 0  # updated after features are loaded
        self.par_map = {}
        self.print_df = pd.DataFrame()
        self.colored_ind_df = pd.DataFrame()
        self.nar_df = pd.DataFrame()
        self.nar_map = {}
        self.db_names = ['merged', 'sent_db', 'sim_vec']
        self.doc_splits = {}
        self.get_doc_splits()

    def get_doc_splits(self):
        for split_idx, split in self.splits.items():
            if self.doc_idx in split['train'] or self.doc_idx in split['test']:
                self.doc_splits[split_idx] = split

    def load_doc_features(self):
        self.doc_db['merged'] = pd.read_csv(os.path.join(
            self.path, "{:02d}_{}.csv".format(self.doc_idx, self.merged_str)))
        self.doc_db['sent_db'] = pd.read_csv(os.path.join(
            self.path, "{:02d}_sent_db.csv".format(self.doc_idx)), usecols=['text', 'par_type', 'nar_idx'])
        self.doc_db['sim_vec'] = pd.read_csv(os.path.join(
            self.path, "{:02d}_sent_sim_vec300_db.csv".format(self.doc_idx)))
        self.doc_len = self.doc_db['merged'].shape[0]
        for split_idx, split in self.doc_splits.items():
            if not split_idx in self.doc_db:
                self.doc_db[split_idx] = {}
            for tf_key, tf_item in self.tf_params[split_idx].items():
                db_key = 'tfidf_{}'.format(tf_item.tf_type)
                # self.doc_db[split_idx][db_key] = sparse.load_npz(os.path.join(
                #         self.path, "{:02d}_{}_tfidf_{}{}.npz".format(self.doc_idx, split_idx, tf_item.tf_type, tf_item.suffix)))
                file_name = "{:02d}_{}_tfidf_{}{}.npz".format(
                    self.doc_idx, split_idx, tf_item.tf_type, tf_item.suffix)
                self.doc_db[split_idx][db_key] = common_utils.open_sparse(
                    self.path, file_name)
        for name in self.db_names:
            feature_utils.curr_doc_db[name] = self.doc_db[name]

    def pack_doc_features(self, tf_to_use):
        for sent_idx in range(self.doc_len):
            par_idx = self.doc_db['merged'].loc[sent_idx, 'par_idx_in_doc']
            sent = Sentence(self.doc_idx, self.doc_len, par_idx,
                            sent_idx, self.doc_db['sent_db'].loc[sent_idx, 'text'],
                            self.doc_db['sent_db'].loc[sent_idx, 'par_type'],
                            self.doc_db['sent_db'].loc[sent_idx, 'nar_idx'])
            # set features per split
            for split_idx, split in self.doc_splits.items():
                for tf_key, tf_item in self.tf_params[split_idx].items():
                    db_key = 'tfidf_{}'.format(tf_item.tf_type)
                    feature_utils.curr_doc_db[db_key] = self.doc_db[split_idx][db_key]
                sent.set_features(feature_utils.sent2features(
                    sent_idx=sent_idx,
                    idx_in_seq=sent_idx,
                    seq_len=self.doc_len,
                    neighbor_radius=self.neighbor_radius,
                    tf_to_use=tf_to_use,
                    tf_features=self.tf_params[split_idx]),
                    split_idx)
            sent.set_y(feature_utils.sent2label(sent_idx))
            if sent.get_y() == 'is_nar':
                self.assign_sentence_to_narative(sent.nar_idx, sent.text)
            self.sent_list.append(sent)

    def assign_sentence_to_narative(self, nar_idx, text):
        if not nar_idx in self.nar_map:
            self.nar_map[nar_idx] = ""
        self.nar_map[nar_idx] += text+". "

    def get_nar_df(self):
        self.nar_df = pd.DataFrame.from_dict(self.nar_map, orient='index')
        return self.nar_df

    def remove_dbs(self):
        del self.doc_db

    def pack_doc(self, tf_to_use=['lemma', 'word', 'char_wb']):
        print('tf to be used: {}'.format(tf_to_use))
        self.load_doc_features()
        self.pack_doc_features(tf_to_use)
        self.remove_dbs()

    def reshape_doc(self, seq_len, step):
        shape_name = "{}_{}".format(seq_len, step)
        self.reshaped[shape_name] = [self.sent_list[i: i+seq_len]
                                     for i in np.arange(0, len(self.sent_list), step)]
        print("Doc {} reshaped from {} to {}".format(
            self.doc_idx, self.doc_len, len(self.reshaped[shape_name])))

    def get_x(self, reshaped_name='', split_idx=0):
        x = []
        if not reshaped_name:
            x = [sent.get_x(split_idx) for sent in self.sent_list]
        else:
            for sent_seq in self.reshaped[reshaped_name]:
                x.append([sent_seq[i].get_x(split_idx)
                         for i in range(len(sent_seq))])
        return x

    def get_y(self, reshaped_name=''):
        y = []
        if not reshaped_name:
            y = [sent.get_y() for sent in self.sent_list]
        else:
            for sent_seq in self.reshaped[reshaped_name]:
                y.append([sent_seq[i].get_y() for i in range(len(sent_seq))])
        return y

    def get_text(self, reshaped_name=''):
        text = []
        if not reshaped_name:
            text = [sent.get_text() for sent in self.sent_list]
        else:
            for sent_seq in self.reshaped[reshaped_name]:
                text.append([sent_seq[i].get_text()
                            for i in range(len(sent_seq))])
        return text

    def get_group(self, reshaped_name=''):
        group = []
        if not reshaped_name:
            group = [self.doc_idx for sent in self.sent_list]
        else:
            group = [self.doc_idx for seq in self.reshaped[reshaped_name]]
        return group

    def get_paragraph(self, reshaped_name=''):
        par = []
        if not reshaped_name:
            par = [sent.get_paragraph() for sent in self.sent_list]
        else:
            for sent_seq in self.reshaped[reshaped_name]:
                par.append([sent_seq[i].get_paragraph()
                           for i in range(len(sent_seq))])
        return par

    def pack_sent_per_paragraph(self):
        self.par_map = {}
        for sent in self.sent_list:
            if not sent.par_idx in self.par_map:
                self.par_map[sent.par_idx] = Paragraph(
                    self.doc_idx, sent.par_idx, sent.par_type)
            self.par_map[sent.par_idx].add_sent(sent)

    def set_pred_y(self, pred_y):
        if len(pred_y) != len(self.sent_list):
            raise Exception("y predicted {} does not match save sentence number {}".format(
                len(pred_y), len(self.sent_list)))
        for i, sent in enumerate(self.sent_list):
            sent.set_pred_y(pred_y[i])

    def get_pred_y(self, reshaped_name=''):
        y = []
        if not reshaped_name:
            y = [sent.get_pred_y() for sent in self.sent_list]
        else:
            for sent_seq in self.reshaped[reshaped_name]:
                y.append([sent_seq[i].get_pred_y()
                         for i in range(len(sent_seq))])
        return y

    def print(self, label="true"):
        for i, par in self.par_map.items():
            self.print_df.loc[i, label] = par.print(label)
            self.print_df.loc[i, "type"] = par.get_par_type()
        self.write_html(label)
        return self.print_df

    def print_colored_indices(self, y_pred=None):
        if y_pred is None and self.sent_list[0].get_pred_y() is None:
            raise Exception("y predicted was not set!")
        self.colored_ind_df.loc[0, 'true'] = model_utils.get_colored_from_list(
            self.get_y(), "is_nar")
        self.colored_ind_df.loc[0, 'pred'] = model_utils.get_colored_from_list(
            self.get_pred_y(), "is_nar")
        self.write_html("colored_indices")
        return self.colored_ind_df

    def write_html(self, label, index=True, style="df_style"):
        html = self.print_df.to_html(
            escape=False, justify="center", index=index)
        html = r'<link rel="stylesheet" type="text/css" href="{}.css" /><br>'.format(
            style) + html
        # write html to file
        print_df_path = os.path.join(
            self.path, "{:02}_print_{}.html".format(self.doc_idx, label)
        )
        text_file = open(print_df_path, "w")
        text_file.write(html)
        text_file.close()


class Dataset:
    def __init__(self,
                 dir_name=None,
                 merged_str="merged_db",
                 splits={},
                 neighbor_radius=3,
                 doc_indices=[]
                 ):
        self.doc_map = {}
        self.dir_name = dir_name
        self.path = os.path.join(os.getcwd(), defines.PATH_TO_DFS, dir_name)
        self.neighbor_radius = neighbor_radius
        self.splits = splits
        self.merged_str = merged_str
        if len(doc_indices) == 0:
            self.doc_indices = np.arange(1, 81)
        else:
            self.doc_indices = doc_indices
        self.tf_params = {}
        self.nar_df = pd.DataFrame()
        self.print_df = pd.DataFrame()
        print("{} init called".format(self.__class__.__name__))

    def pack_dataset(self, tf_to_use=['lemma', 'word', 'char_wb']):
        print("\nPacking dataset...")

        for idx in self.doc_indices:
            doc = Document(idx=idx,
                           path=self.path,
                           tf_params=self.tf_params,
                           splits=self.splits,
                           merged_str=self.merged_str,
                           neighbor_radius=self.neighbor_radius)
            doc.pack_doc(tf_to_use)
            print("{}".format(idx, end=' '))
            self.doc_map[idx] = doc

    def reshape(self, seq_len, step):
        print("\nReshaping dataset: seq_len {}, step {}...".format(seq_len, step))
        for idx in self.doc_indices:
            self.doc_map[idx].reshape_doc(seq_len, step)
            print("{}".format(idx, end=' '))

    def get_x(self, doc_indices, reshape_name='', split_idx=0):
        x = []
        for idx in doc_indices:
            x.extend(self.doc_map[idx].get_x(reshape_name, split_idx))
        return x

    def get_paragraph(self, doc_indices, reshape_name=''):
        par = []
        for idx in doc_indices:
            par.extend(self.doc_map[idx].get_paragraph(reshape_name))
        return par

    def get_text(self, doc_indices, reshape_name=''):
        text = []
        for idx in doc_indices:
            text.extend(self.doc_map[idx].get_text(reshape_name))
        return text

    def get_y(self, doc_indices, reshape_name=''):
        y = []
        for idx in doc_indices:
            y.extend(self.doc_map[idx].get_y(reshape_name))
        return y

    def get_group(self, doc_indices, reshape_name=''):
        group = []
        for idx in doc_indices:
            group.extend(self.doc_map[idx].get_group(reshape_name))
        return group

    def dump_to_file(self, file_name):
        path = os.path.join(self.path, "dataset_"+file_name+".p")
        # open a file, where you ant to store the data
        file = open(path, 'wb')
        # dump information to that file
        pickle.dump(self, file)
        # close the file
        file.close()

    def set_tf_params(self, tf_name, stop_list=[]):
        for split_idx in self.splits.keys():
            if not split_idx in self.tf_params:
                self.tf_params[split_idx] = {}
            self.tf_params[split_idx][tf_name] = TfParams(
                self.dir_name, tf_name, split_idx, self.splits)

    def pack_sent_per_paragraph(self):
        for idx in self.doc_indices:
            self.doc_map[idx].pack_sent_per_paragraph()

    def print(self, label="true"):
        for idx in self.doc_indices:
            doc_df = self.doc_map[idx].print(label)
            doc_df['doc_idx'] = idx
            self.print_df = pd.concat(
                [self.print_df, doc_df], ignore_index=False, axis=0, copy=False)
        return self.print_df

    def get_nar_df(self):
        for idx in self.doc_indices:
            doc_df = self.doc_map[idx].get_nar_df()
            doc_df['doc_idx'] = idx
            self.nar_df = pd.concat(
                [self.nar_df, doc_df], ignore_index=False, axis=0, copy=False)
        return self.nar_df

    def copy_attr(self, other):
        self.__dict__ = other.__dict__.copy()

    def copy_attr(self, other):
        self.__dict__ = other.__dict__.copy()


class WindowDiff():
    def __init__(self):
        print("{} init called".format(self.__class__.__name__))

    def count_boundaries(self, y, start_idx, end_idx):
        cnt = np.count_nonzero(np.diff(y[start_idx:end_idx]))
        # print("[{}:{}] y={} cnt {}".format(start_idx,end_idx-1,y[start_idx:end_idx],cnt))
        return cnt

    def get_boundaries_db(self, y):
        start, end = self.get_boundaries_indices(y)
        bound_db = pd.DataFrame()
        bound_db['start'] = start
        bound_db['end'] = end
        return bound_db

    def get_boundaries_indices(self, y):
        y, _ = self.convert_labels(y, None)
        y_str = ''.join(str(item) for item in y)
        start_str = '01'
        end_str = '10'
        nar_begin = []
        nar_end = []
        if re.match(r'^1', y_str):
            nar_begin.append(0)
        for match in re.finditer(start_str, y_str):
            nar_begin.append(match.start()+1)
        for match in re.finditer(end_str, y_str):
            nar_end.append(match.start()+1)
        if re.search(r'1$', y_str):
            nar_end.append(len(y_str))

        return nar_begin, nar_end

    def get_near_miss_idx(self, y_true, y_pred):
        misses = {}
        misses['start'] = {}
        misses['end'] = {}
        misses['start']['fp-1'] = []
        misses['start']['fn+1'] = []
        misses['end']['fn-1'] = []
        misses['end']['fp+1'] = []

        for start_idx in y_true['start']:
            if (start_idx+1 in y_pred['start']):
                misses['start']['fn+1'].append(start_idx)
            if (start_idx-1 in y_pred['start']):
                misses['start']['fp-1'].append(start_idx-1)
        for end_idx in y_true['end']:
            if (end_idx+1 in y_pred['end']):
                misses['end']['fp+1'].append(end_idx+1)
            if (end_idx-1 in y_pred['end']):
                misses['end']['fn-1'].append(end_idx)
        return misses

    def convert_labels(self, y_true, y_pred):
        if isinstance(y_true[0], str):
            y_true = common_utils.convert_str_label_to_binary(y_true)
            if y_pred:
                y_pred = common_utils.convert_str_label_to_binary(y_pred)
        return y_true, y_pred

    def calc_penalty(self, y_true, y_pred, window_size=15):
        if(len(y_true) != len(y_pred)):
            raise Exception("y true and predicted length doesn't match")
        penalty = 0
        y_len = len(y_true)
        y_true, y_pred = self.convert_labels(y_true, y_pred)
        true_count = self.count_boundaries(y_true, 0, y_len)
        pred_count = self.count_boundaries(y_pred, 0, y_len)
        num_of_steps = y_len-window_size
        for start_idx in range(num_of_steps):
            end_idx = start_idx+window_size
            true_count = self.count_boundaries(y_true, start_idx, end_idx)
            pred_count = self.count_boundaries(y_pred, start_idx, end_idx)
            wind_diff = abs(true_count-pred_count)
            penalty += wind_diff
            # print("[{}:{}]={} true count {} prd count {} diff {} penalty {}".format(start_idx, end_idx-1, y_true[start_idx:end_idx],
            #   true_count, pred_count, wind_diff, penalty))
        return penalty/(num_of_steps), y_len, true_count, pred_count


class WinPR(WindowDiff):
    def __init__(self, window_size=3):
        self.window_size = window_size
        print("{} init called".format(self.__class__.__name__))

    def count_boundaries(self, y, start_idx, end_idx):
        start_idx_pad = max(0, start_idx-1)
        end_idx_pad = min(end_idx+1, len(y)-1)
        return super().count_boundaries(y, start_idx_pad, end_idx_pad)

    def padd_y(self, y, padd_len):
        return [y[0] for i in range(padd_len)] + y + [y[-1] for i in range(padd_len)]

    def calc_errors(self, y_true, y_pred):
        if(len(y_true) != len(y_pred)):
            raise Exception("y true and predicted length doesn't match")
        y_true, y_pred = super().convert_labels(y_true, y_pred)
        y_true = self.padd_y(y_true, self.window_size-1)
        y_pred = self.padd_y(y_pred, self.window_size-1)
        # print("len padded {}".format(len(y_pred)))
        result = {}
        bound_candidate_num = self.window_size+1
        result['tp'] = 0
        result['tn'] = -bound_candidate_num*self.window_size
        result['fp'] = 0
        result['fn'] = 0
        for start_idx in range(len(y_pred)-self.window_size+1):
            end_idx = start_idx+self.window_size
            true_count = self.count_boundaries(y_true, start_idx, end_idx)
            pred_count = self.count_boundaries(y_pred, start_idx, end_idx)
            result['tp'] += min(true_count, pred_count)
            result['tn'] += (bound_candidate_num-max(true_count, pred_count))
            result['fp'] += max(0, pred_count-true_count)
            result['fn'] += max(0, true_count-pred_count)
        for key, val in result.items():
            result[key] = val/bound_candidate_num

        return result

    def f_score(self, result):
        if result['tp'] == 0:
            precision = 0
            recall = 0
            f1 = recall
        else:
            precision = result['tp']/(result['tp']+result['fp'])
            recall = result['tp']/(result['tp']+result['fn'])
            denom = precision + recall
            f1 = 2*(precision*recall)/denom
        return precision, recall, f1

    def get_score(self, y_true, y_pred):
        result = self.calc_errors(y_true, y_pred)
        p, r, f1 = self.f_score(result)
        return p, r, f1

    def score_func(self, y_true, y_pred):
        if isinstance(y_true[0], list):
            y_true = flatten(y_true)
            y_pred = flatten(y_pred)
        _, _, f1 = self.get_score(y_true, y_pred)
        return f1


class Split:
    def __init__(self, train, test):
        print("{} init called".format(self.__class__.__name__))
        self.train = train
        self.test = test


class MyMixedScorer:
    def __init__(self, window_size=3, weights=[0.5, 0.5]):
        print("{} init called".format(self.__class__.__name__))
        self.win_f1 = WinPR(window_size=window_size)
        self.weights = weights

    def score_func(self, y_true, y_pred):
        win_score = self.win_f1.score_func(y_true, y_pred)
        label_score = metrics.flat_f1_score(y_true, y_pred)
        print("winPR {} label {}".format(win_score, label_score))
        mixed_score = np.average(
            a=[win_score, label_score], weights=self.weights)
        return mixed_score


class MySegEval():
    def __init__(self, main_score='f1', n_t=2):
        self.main_score = main_score
        self.n_t = n_t
        self.scores = {}
        print("{} init called".format(self.__class__.__name__))

    def get_segment_list(self, y):
        df = pd.DataFrame(y, columns=['is_nar'])
        seg_size = pd.DataFrame()
        seg_size['size'] = df.groupby(
            (df['is_nar'].shift() != df['is_nar']).cumsum()).size()
        seg_size['is_nar'] = df.groupby((df['is_nar'].shift() != df['is_nar']).cumsum())[
            'is_nar'].apply(common_utils.get_single_unique)
        return seg_size['size'].tolist()

    def score_func(self, y_true, y_pred):
        scores = self.get_scores(y_true, y_pred)
        return scores[self.main_score]

    def flatten_y(self, y):
        if isinstance(y[0], list):
            y = flatten(y)
        return y

    def get_scores(self, y_true, y_pred):
        labels = np.unique(y_true)
        y_true = self.get_segment_list(self.flatten_y(y_true))
        y_pred = self.get_segment_list(self.flatten_y(y_pred))
        conf_matrix = se.boundary_confusion_matrix(
            y_true, y_pred, n_t=self.n_t)
        self.scores['f1'] = se.fmeasure(conf_matrix)
        self.scores['recall'] = se.recall(conf_matrix)
        self.scores['precision'] = se.recall(conf_matrix)
        self.scores['b_sim'] = se.boundary_similarity(
            y_true, y_pred, n_t=self.n_t)
        self.scores['s_sim'] = se.segmentation_similarity(
            y_true, y_pred, n_t=self.n_t)
        self.scores['b_stat'] = se.boundary_statistics(
            y_true, y_pred, n_t=self.n_t)
        return self.scores


class MyScoreSummarizer():

    def __init__(self, pred_df, fix_list, prefixes=['bert', 'crf', 'ens'], score='weighted'):
        print("{} init called".format(self.__class__.__name__))
        self.pred_df = pred_df.copy()
        self.convert_labels_to_int()
        self.report = {}
        self.print_df = {}
        self.latex_str = {}
        self.fix_list = fix_list
        if len(self.fix_list) > 0:
            self.my_fixer = MyPredFixer(self.pred_df, self.fix_list, prefixes)
            self.my_fixer.fix_error_prefixes(prefixes)
            self.my_fixer.get_stat_for_prefixes(prefixes)
            self.fixed_df = self.my_fixer.fixed_df
        self.prefixes = prefixes
        self.f_s_scores = {}
        self.f_s_dict = {}
        self.par_scores = {}
        self.par_dict = {}
        self.f_par_scores = {}
        self.f_par_dict = {}
        self.s_scores = {}
        self.s_dict = {}

    def convert_labels_to_int(self):
        self.pred_df.replace({'not_nar': 0, 'is_nar': 1}, inplace=True)

    def get_score(self, prefix, unit='split'):
        fixed_dict = {}
        self.labels = [
            str(i) for i in self.pred_df['{}_true'.format(prefix)].unique().tolist()]
        self.s_scores[prefix], self.s_dict[prefix], self.par_scores[prefix], self.par_dict[prefix] = model_utils.get_report_by_unit(
            self.pred_df, prefix, unit, print_rep=False, segeval=False, use_par=False)
        if len(self.fix_list) > 0:
            self.f_s_scores[prefix], self.f_s_dict[prefix], self.f_par_scores[prefix], self.f_par_dict[prefix] = model_utils.get_report_by_unit(
                self.fixed_df, prefix, unit, print_rep=False, segeval=False, use_par=False)
            fixed_dict = self.f_s_dict[prefix]
        self.report[prefix] = MyReport(
            self.s_dict[prefix], fixed_dict, self.par_dict[prefix], prefix, labels=self.labels)
        self.print_df[prefix] = self.report[prefix].get_print_df()
        self.get_latex_table(prefix)

    def get_latex_table(self, prefix):
        self.latex_str[prefix] = self.print_df[prefix].to_csv(
            sep='&', line_terminator="\\", float_format='%.3f')

    def get_all_scores(self, unit='split'):
        for t in self.prefixes:
            self.get_score(t, unit)


class MyReport():

    def __init__(self, sent_dict, fixed_dict, par_dict, name, labels=['not_nar', 'nar']):
        print("{} init called".format(self.__class__.__name__))
        if not (isinstance(sent_dict, dict) and isinstance(par_dict, dict)):
            raise Exception("Expect to get dictionary as input")
        self.scr_dict = {}
        self.types = ['sent']
        if fixed_dict:
            self.types.append('fixed')
            self.scr_dict['fixed'] = fixed_dict
        self.scr_dict['sent'] = sent_dict
        if par_dict:
            self.types.append('par')
            self.scr_dict['par'] = par_dict
        self.avg = {}
        self.labels = []
        for l in labels:
            self.labels.append(l)
        self.labels.append('weighted avg')

        self.metrics = ['f1', 'recall', 'prec']
        self.name = name

    def get_avg_scores(self, name):
        self.avg[name] = pd.DataFrame()

        for label in self.labels:
            for key, val in self.scr_dict[name].items():
                self.avg[name].loc[key, '{}_f1'.format(
                    label)] = val[label]['f1-score']
                self.avg[name].loc[key, '{}_recall'.format(
                    label)] = val[label]['recall']
                self.avg[name].loc[key, '{}_prec'.format(
                    label)] = val[label]['precision']

        self.avg[name].loc['mean'] = self.avg[name].mean()

        return self.avg[name]

    def get_print_df(self):
        for t in self.types:
            self.get_avg_scores(t)
        self.print_df = pd.DataFrame()
        for m in self.metrics:
            for l in self.labels:
                for t in self.types:
                    self.print_df.loc[l, '{}_{}'.format(
                        t, m)] = self.avg[t].loc['mean', "{}_{}".format(l, m)]
        return self.print_df


class MyScorer():

    def __init__(self):
        print("{} init called".format(self.__class__.__name__))
        self.custom_scorer = {'accuracy': make_scorer(accuracy_score),
                              'balanced_accuracy': make_scorer(balanced_accuracy_score),
                              'precision': make_scorer(precision_score, average='weighted'),
                              'recall': make_scorer(recall_score, average='weighted'),
                              'f1': make_scorer(f1_score, average='weighted'),
                              }
        self.scorer_names = list(self.custom_scorer.keys())
        self.scores_df = pd.DataFrame()

    def add_score(self, scores, regressorName, prefix):
        for name in self.scorer_names:
            self.scores_df.loc[regressorName + '_' +
                               prefix, name] = scores["test_"+name].mean()

    def get_cross_val_score(self, estimator, X_train, y_train, groups, prefix="", cv=10):
        name = estimator.__class__.__name__
        start_time = time.time()
        print('*********' + name + '*********')
        full_scores = cross_validate(
            estimator,
            X_train,
            y_train,
            groups=groups,
            cv=cv,
            scoring=self.custom_scorer,
            n_jobs=-1
        )
        self.add_score(full_scores, name, prefix)
        end_time = time.time()-start_time
        print("{} took  {}".format(name, time.strftime(
            "%H:%M:%S", time.gmtime(end_time))))


class MyPredFixer():
    def __init__(self, pred_df, fix_list, prefixes):
        print("{} init called".format(self.__class__.__name__))
        self.wd = WindowDiff()
        self.pred_df = pred_df
        self.orig_df = self.pred_df.copy()
        self.fix_list = fix_list
        self.fixed_df = self.pred_df.copy()
        t = prefixes[0]
        self.true_nar_idx = self.orig_df.query(
            '{}_true == 1'.format(t)).index.tolist()
        self.true_nn_idx = self.orig_df.query(
            '{}_true == 0'.format(t)).index.tolist()
        self.prefixes = prefixes
        self.stat = {}
        self.near_misses = {}
        self.middle_miss = {}
        self.stand_alone = {}
        self.double_stand_alone = {}
        self.double_middle_miss = {}
        self.tree_middle_miss = {}
        self.four_middle_miss = {}
        for t in self.prefixes:
            self.stat[t] = {}
            self.count_pos_pred(t)

    def count_pos_pred(self, prefix):
        self.stat[prefix]['pos_pred_0.5'] = self.pred_df['{}_predicted'.format(
            prefix)].sum()

    def fix_proba(self, prefix, threshold):
        if not '{}_proba_1'.format(prefix) in self.pred_df.columns:
            raise Exception("No proba column for {}".format(prefix))
        print("Total pos {} {} before using threshold {}".format(
            self.stat[prefix]['pos_pred_0.5'], prefix, threshold))
        self.fixed_df['{}_predicted'.format(prefix)] = self.fixed_df['{}_proba_1'.format(
            prefix)].apply(lambda x: 1 if x > threshold else 0)
        print("Total pos {} {} after  using threshold {}".format(
            self.fixed_df['{}_predicted'.format(prefix)].sum(), prefix, threshold))

    def get_near_miss(self, prefix):
        start_true, end_true = self.wd.get_boundaries_indices(
            self.pred_df['{}_true'.format(prefix)].tolist())
        start_pred, end_pred = self.wd.get_boundaries_indices(
            self.pred_df['{}_predicted'.format(prefix)].tolist())
        self.near_misses[prefix] = self.wd.get_near_miss_idx(
            y_true={'start': start_true, 'end': end_true}, y_pred={'start': start_pred, 'end': end_pred})
        self.get_near_miss_stat(prefix)

    def get_near_miss_stat(self, prefix):
        if prefix not in self.near_misses:
            self.get_near_miss(prefix)
        self.stat[prefix]['near'] = {}
        self.stat[prefix]['near']['tot'] = 0
        self.stat[prefix]['near']['fp'] = 0
        self.stat[prefix]['near']['fn'] = 0
        for k, v in self.near_misses[prefix].items():
            for i, j in v.items():
                self.stat[prefix]['near']['tot'] += len(j)
                if 'fp' in i:
                    self.stat[prefix]['near']['fp'] += len(j)
                if 'fn' in i:
                    self.stat[prefix]['near']['fn'] += len(j)

    def get_mid_miss_stat(self, prefix):
        if prefix not in self.middle_miss:
            self.get_middle_miss(prefix)
        self.stat[prefix]['mid'] = len(self.middle_miss[prefix])
        miss_orig = set(self.true_nar_idx) & set(self.middle_miss[prefix])
        self.stat[prefix]['mid_orig'] = len(miss_orig)

    def get_double_mid_miss_stat(self, prefix):
        self.stat[prefix]['double_mid_miss'] = len(
            self.double_middle_miss[prefix])
        miss_orig = set(self.true_nar_idx) & set(
            self.double_middle_miss[prefix])
        self.stat[prefix]['double_mid_miss_orig'] = len(miss_orig)

    def get_stand_alone_stat(self, prefix):
        if prefix not in self.stand_alone:
            self.get_stand_alone(prefix)
        self.stat[prefix]['stand_alone'] = len(self.stand_alone[prefix])
        miss_orig = set(self.true_nn_idx) & set(self.stand_alone[prefix])
        self.stat[prefix]['stand_alone_orig'] = len(miss_orig)

    def get_double_stand_alone_stat(self, prefix):
        self.stat[prefix]['double_stand_alone'] = len(
            self.double_stand_alone[prefix])
        miss_orig = set(self.true_nn_idx) & set(
            self.double_stand_alone[prefix])
        self.stat[prefix]['double_stand_alone_orig'] = len(miss_orig)

    def get_tree_mid_miss_stat(self, prefix):
        self.stat[prefix]['tree_mid_miss'] = len(
            self.tree_middle_miss[prefix])
        miss_orig = set(self.true_nar_idx) & set(self.tree_middle_miss[prefix])
        self.stat[prefix]['tree_mid_miss_orig'] = len(miss_orig)

    def get_four_mid_miss_stat(self, prefix):
        self.stat[prefix]['four_mid_miss'] = len(
            self.four_middle_miss[prefix])
        miss_orig = set(self.true_nar_idx) & set(self.four_middle_miss[prefix])
        self.stat[prefix]['four_middle_miss_orig'] = len(miss_orig)

    def get_total_stat(self, prefix):
        self.stat[prefix]['total_fp'] = self.pred_df[(self.pred_df['{}_true'.format(prefix)] == 0) & (
            self.pred_df['{}_predicted'.format(prefix)] == 1)].shape[0]
        self.stat[prefix]['total_fn'] = self.pred_df[(self.pred_df['{}_true'.format(prefix)] == 1) & (
            self.pred_df['{}_predicted'.format(prefix)] == 0)].shape[0]

    def get_all_stat(self, prefix):
        self.get_total_stat(prefix)
        self.get_near_miss_stat(prefix)
        if('single_miss' in self.fix_list):
            self.get_mid_miss_stat(prefix)
        if('single_sa' in self.fix_list):
            self.get_stand_alone_stat(prefix)
        if('double_miss' in self.fix_list):
            self.get_double_mid_miss_stat(prefix)
        if('tree_miss' in self.fix_list):
            self.get_tree_mid_miss_stat(prefix)
        if('four_miss' in self.fix_list):
            self.get_four_mid_miss_stat(prefix)
        if('double_sa' in self.fix_list):
            self.get_double_stand_alone_stat(prefix)

    def fix_near_miss(self, prefix):
        self.get_near_miss(prefix)
        print("near misses to be fixed\n", self.stat[prefix]['near'])
        self.fixed_df.loc[self.near_misses[prefix]['start']
                          ['fp-1'], '{}_predicted'.format(prefix)] = 0
        self.fixed_df.loc[self.near_misses[prefix]['start']
                          ['fn+1'], '{}_predicted'.format(prefix)] = 1
        self.fixed_df.loc[self.near_misses[prefix]['end']
                          ['fp+1'], '{}_predicted'.format(prefix)] = 0
        self.fixed_df.loc[self.near_misses[prefix]['end']
                          ['fn-1'], '{}_predicted'.format(prefix)] = 1

    def get_middle_miss(self, prefix):
        # true_narr = self.pred_df[self.pred_df['{}_true'.format(
        #     prefix)] == 1].copy()
        middle_miss_df = self.fixed_df['{}_predicted'.format(prefix)].where(
            (self.fixed_df['{}_predicted'.format(prefix)] == 0) &
            (self.fixed_df['{}_predicted'.format(prefix)].shift(1) == 1) &
            (self.fixed_df['{}_predicted'.format(prefix)].shift(-1) == 1)
        )
        # middle_miss_df = self.fixed_df['{}_predicted'.format(prefix)].where(
        #     (self.fixed_df['{}_predicted'.format(prefix)] == 0) &
        #     (self.fixed_df['{}_predicted'.format(prefix)].shift(1) == 1) &
        #     (true_narr['{}_predicted'.format(prefix)].shift(-1) == 1)
        #     )
        middle_miss_df.dropna(inplace=True)
        self.middle_miss[prefix] = middle_miss_df.index.tolist()
        self.get_mid_miss_stat(prefix)

    def get_double_middle_miss(self, prefix):
        # true_narr = self.fixed_df[self.fixed_df['{}_true'.format(
        #     prefix)] == 1].copy()
        double_middle_miss_df = self.fixed_df['{}_predicted'.format(prefix)].where(
            (self.fixed_df['{}_predicted'.format(prefix)] == 0) &
            (self.fixed_df['{}_predicted'.format(prefix)].shift(1) == 0) &
            (self.fixed_df['{}_predicted'.format(prefix)].shift(-1) == 1) &
            (self.fixed_df['{}_predicted'.format(prefix)].shift(2) == 1)
            # &
            # (self.fixed_df['{}_predicted'.format(prefix)].shift(-2) == 1) &
            # (self.fixed_df['{}_predicted'.format(prefix)].shift(3) == 1)
        )
        double_middle_miss_df.dropna(inplace=True)
        self.double_middle_miss[prefix] = double_middle_miss_df.index.tolist()
        orig = np.array(self.double_middle_miss[prefix].copy())
        for i in range(1, 2):
            plus = orig+i
            self.double_middle_miss[prefix].extend(plus)
        self.get_double_mid_miss_stat(prefix)

    def get_tree_middle_miss(self, prefix):
        # true_narr = self.fixed_df[self.fixed_df['{}_true'.format(
        #     prefix)] == 1].copy()
        tree_middle_miss_df = self.fixed_df['{}_predicted'.format(prefix)].where(
            (self.fixed_df['{}_predicted'.format(prefix)] == 0) &
            (self.fixed_df['{}_predicted'.format(prefix)].shift(1) == 0) &
            (self.fixed_df['{}_predicted'.format(prefix)].shift(2) == 0) &
            (self.fixed_df['{}_predicted'.format(prefix)].shift(-1) == 1) &
            (self.fixed_df['{}_predicted'.format(prefix)].shift(3) == 1)
            # &
            # (self.fixed_df['{}_predicted'.format(prefix)].shift(3) == 1) &
            # (self.fixed_df['{}_predicted'.format(prefix)].shift(-3) == 1)
        )
        tree_middle_miss_df.dropna(inplace=True)
        self.tree_middle_miss[prefix] = tree_middle_miss_df.index.tolist()
        orig = np.array(self.tree_middle_miss[prefix].copy())
        for i in range(1, 3):
            plus = orig + i
            self.tree_middle_miss[prefix].extend(plus)
        self.get_tree_mid_miss_stat(prefix)

    def get_four_middle_miss(self, prefix):
        # true_narr = self.fixed_df[self.fixed_df['{}_true'.format(
        #     prefix)] == 1].copy()
        four_middle_miss_df = self.fixed_df['{}_predicted'.format(prefix)].where(
            (self.fixed_df['{}_predicted'.format(prefix)] == 0) &
            (self.fixed_df['{}_predicted'.format(prefix)].shift(1) == 0) &
            (self.fixed_df['{}_predicted'.format(prefix)].shift(2) == 0) &
            (self.fixed_df['{}_predicted'.format(prefix)].shift(3) == 0) &
            (self.fixed_df['{}_predicted'.format(prefix)].shift(-1) == 1) &
            (self.fixed_df['{}_predicted'.format(prefix)].shift(4) == 1)
        )
        four_middle_miss_df.dropna(inplace=True)
        self.four_middle_miss[prefix] = four_middle_miss_df.index.tolist()
        orig = np.array(self.four_middle_miss[prefix].copy())
        for i in range(1, 4):
            plus = orig + i
            self.four_middle_miss[prefix].extend(plus)
        self.get_four_mid_miss_stat(prefix)

    def fix_double_middle_miss(self, prefix):
        self.get_double_middle_miss(prefix)
        print("{} double middle misses to be fixed".format(
            self.stat[prefix]['double_mid_miss']))
        self.fixed_df.loc[self.double_middle_miss[prefix],
                          '{}_predicted'.format(prefix)] = 1

    def fix_tree_middle_miss(self, prefix):
        self.get_tree_middle_miss(prefix)
        print("{} tree middle misses to be fixed".format(
            self.stat[prefix]['tree_mid_miss']))
        self.fixed_df.loc[self.tree_middle_miss[prefix],
                          '{}_predicted'.format(prefix)] = 1

    def fix_four_middle_miss(self, prefix):
        self.get_four_middle_miss(prefix)
        print("{} four middle misses to be fixed".format(
            self.stat[prefix]['four_mid_miss']))
        self.fixed_df.loc[self.four_middle_miss[prefix],
                          '{}_predicted'.format(prefix)] = 1

    def fix_middle_miss(self, prefix):
        self.get_middle_miss(prefix)
        print("{} middle misses to be fixed".format(self.stat[prefix]['mid']))
        self.fixed_df.loc[self.middle_miss[prefix],
                          '{}_predicted'.format(prefix)] = 1

    def get_stand_alone(self, prefix):
        # true_not_nar = self.fixed_df[self.fixed_df['{}_true'.format(
        #     prefix)] == 0].copy()
        stand_alone_df = self.fixed_df['{}_predicted'.format(prefix)].where(((self.fixed_df['{}_predicted'.format(prefix)] == 1) &
                                                                            (self.fixed_df['{}_predicted'.format(prefix)].shift(1) == 0) &
                                                                            (self.fixed_df['{}_predicted'.format(prefix)].shift(-1) == 0) &
                                                                            (self.fixed_df['{}_predicted'.format(prefix)].shift(2) == 0) &
                                                                            (self.fixed_df['{}_predicted'.format(prefix)].shift(-2) == 0) &
                                                                            (self.fixed_df['{}_predicted'.format(prefix)].shift(3) == 0) &
                                                                            (self.fixed_df['{}_predicted'.format(prefix)].shift(-3) == 0) &
                                                                            (self.fixed_df['{}_predicted'.format(prefix)].shift(4) == 0) &
                                                                            (self.fixed_df['{}_predicted'.format(
                                                                                prefix)].shift(-4) == 0)
                                                                             ))
        stand_alone_df.dropna(inplace=True)
        self.stand_alone[prefix] = stand_alone_df.index.tolist()
        self.get_stand_alone_stat(prefix)

    def get_double_stand_alone(self, prefix):
        # true_not_nar = self.fixed_df[self.fixed_df['{}_true'.format(
        #     prefix)] == 0].copy()
        double_stand_alone_df = self.fixed_df['{}_predicted'.format(prefix)].where((self.fixed_df['{}_predicted'.format(prefix)] == 1) &
                                                                                   (self.fixed_df['{}_predicted'.format(prefix)].shift(1) == 1) &
                                                                                   (self.fixed_df['{}_predicted'.format(prefix)].shift(-1) == 0) &
                                                                                   (self.fixed_df['{}_predicted'.format(prefix)].shift(-2) == 0) &
                                                                                   (self.fixed_df['{}_predicted'.format(prefix)].shift(-3) == 0) &
                                                                                   (self.fixed_df['{}_predicted'.format(prefix)].shift(-4) == 0) &
                                                                                   (self.fixed_df['{}_predicted'.format(prefix)].shift(2) == 0) &
                                                                                   (self.fixed_df['{}_predicted'.format(prefix)].shift(3) == 0) &
                                                                                   (self.fixed_df['{}_predicted'.format(prefix)].shift(4) == 0) &
                                                                                   (self.fixed_df['{}_predicted'.format(
                                                                                       prefix)].shift(5) == 0)
                                                                                   )
        double_stand_alone_df.dropna(inplace=True)
        self.double_stand_alone[prefix] = double_stand_alone_df.index.tolist()
        orig = np.array(self.double_stand_alone[prefix].copy())
        for i in range(1, 2):
            plus = orig+i
            self.double_stand_alone[prefix].extend(plus)
        self.get_double_stand_alone_stat(prefix)

    def fix_stand_alone(self, prefix):
        self.get_stand_alone(prefix)
        print("{} stande alone to be fixed".format(
            self.stat[prefix]['stand_alone']))
        self.fixed_df.loc[self.stand_alone[prefix],
                          '{}_predicted'.format(prefix)] = 0

    def fix_double_stand_alone(self, prefix):
        self.get_double_stand_alone(prefix)
        print("{} double stande alone to be fixed".format(
            self.stat[prefix]['double_stand_alone']))
        self.fixed_df.loc[self.double_stand_alone[prefix],
                          '{}_predicted'.format(prefix)] = 0

    def fix_errors(self, prefix):
        self.labels = self.pred_df['{}_true'.format(prefix)].unique().tolist()

        loop = [s for s in self.fix_list if "loop" in s]
        if loop:
            iterr = int(loop[0].split('_')[-1])
            for i in range(iterr):
                self.fix_middle_miss(prefix)
        for mode in self.fix_list:
            if 'threshold' in mode:
                val = float(mode.split('_')[-1])
                self.fix_proba(prefix, val)
            if mode == 'near_miss':
                self.fix_near_miss(prefix)
            if mode == 'four_miss':
                self.fix_four_middle_miss(prefix)
            if mode == 'tree_miss':
                self.fix_tree_middle_miss(prefix)
            if mode == 'double_miss':
                self.fix_double_middle_miss(prefix)
            if mode == 'single_miss':
                self.fix_middle_miss(prefix)
            if mode == 'double_sa':
                self.fix_double_stand_alone(prefix)
            if mode == 'single_sa':
                self.fix_stand_alone(prefix)

    def fix_error_prefixes(self, prefixes):
        for t in prefixes:
            self.fix_errors(t)
            self.get_all_stat(t)

    def get_stat_for_prefixes(self, prefixes):
        if 'bert' in prefixes and 'crf' in prefixes:
            self.get_unique_mistakes()
        for t in prefixes:
            print(t)
            self.get_all_stat(t)

    def get_unique_mistakes(self):
        types = {'bert': 0, 'crf': 0}
        pos_label = 1
        mistakes = {}
        for t in types.keys():
            mistakes[t] = {}
            l = types[t]
            mistakes[t]['total_index'] = set(self.pred_df.query(
                '{}_predicted!={}_true'.format(t, t)).index)
            mistakes[t]['false_neg_index'] = set(self.pred_df.query(
                '{}_predicted!={}_true and {}_predicted==@l'.format(t, t, t)).index)
            mistakes[t]['false_pos_index'] = set(self.pred_df.query(
                '{}_predicted!={}_true and {}_predicted==@pos_label'.format(t, t, t)).index)
            mistakes[t]['total_count'] = len(mistakes[t]['total_index'])
            mistakes[t]['false_neg_count'] = len(
                mistakes[t]['false_neg_index'])
            mistakes[t]['false_pos_count'] = len(
                mistakes[t]['false_pos_index'])

        self.mistakes = mistakes  # store indices of mistakes
        self.stat['bert']['unique'] = len(
            mistakes['bert']['total_index']-mistakes['crf']['total_index'])
        self.stat['crf']['unique'] = len(
            mistakes['crf']['total_index']-mistakes['bert']['total_index'])

        self.stat['bert']['false_neg_unique'] = len(
            mistakes['bert']['false_neg_index']-mistakes['crf']['false_neg_index'])
        self.stat['crf']['false_neg_unique'] = len(
            mistakes['crf']['false_neg_index']-mistakes['bert']['false_neg_index'])

        self.stat['bert']['false_pos_unique'] = len(
            mistakes['bert']['false_pos_index']-mistakes['crf']['false_pos_index'])
        self.stat['crf']['false_pos_unique'] = len(
            mistakes['crf']['false_pos_index']-mistakes['bert']['false_pos_index'])

        self.stat['crf']['common_neg'] = len(
            mistakes['crf']['false_neg_index'] & mistakes['bert']['false_neg_index'])
        self.stat['crf']['common_pos'] = len(
            mistakes['crf']['false_pos_index'] & mistakes['bert']['false_pos_index'])


class MyFeatureSelector():

    def __init__(self, dir_name, dataset, seq_len, step, prefix):
        self.local_doc_map = {}
        self.seq_len = seq_len
        self.step = step
        self.dataset = dataset
        self.dir_name = dir_name
        self.feature_names = ['lemma', 'word', 'char']
        self.stored_f_name = 'local_doc_map'
        self.prefix = prefix
        self.store_features()

    def store_features(self):
        for doc_idx, doc in self.dataset.doc_map.items():
            print(doc_idx, end=' ')
            self.local_doc_map[doc_idx] = []
            for sent in doc.sent_list:
                local_sent = {}
                for split, v in sent.x.items():
                    local_sent[split] = {}
                    for f_name, f_val in v.items():
                        local_sent[split][f_name] = f_val
                self.local_doc_map[doc_idx].append(local_sent)
        common_utils.dump_to_file(
            self.local_doc_map, self.dir_name, self.stored_f_name+self.prefix)

    def remove_features(self, f_left):
        if len(f_left) == len(self.feature_names):
            return
        f_remove = [f for f in self.feature_names if f not in f_left]
        print('features left: ', f_left, 'features removed: ', f_remove)
        self.local_doc_map = common_utils.load_pickle(
            self.dir_name, self.stored_f_name+self.prefix)
        for doc_idx, doc in self.dataset.doc_map.items():
            print(doc_idx, end=' ')
            for sent_idx, sent in enumerate(doc.sent_list):
                for split_idx, split_val in self.local_doc_map[doc_idx][sent_idx].items():
                    new_x = {}
                    for k in split_val.keys():
                        if any(f in k for f in f_remove):
                            continue
                        else:
                            new_x[k] = split_val[k]
                    sent.x[split_idx] = copy.deepcopy(new_x)
        self.check_features(f_left)

    def get_report(self):
        self.crf_report = MyScoreSummarizer(
            pred_df=self.crf_res_db, fix_list=[], prefixes=['crf'])
        self.crf_report.get_all_scores('split')
        self.crf_report.print_df['crf'].index.name = 'label'

    def save_report(self, f_left):
        suffix = '.'.join(f_left)
        common_utils.save_db(self.crf_report.print_df['crf'], self.dir_name, 'crf.split.report.tf.per.split{}.{}'.format(
            suffix, self.prefix), keep_index=True, float_format='%.3f')
        common_utils.save_db(self.crf_res_db, self.dir_name, 'crf.pred.tf.per.split{}.{}'.format(
            suffix, self.prefix), keep_index=True, float_format='%.3f')
        common_utils.save_db(self.crf_f_db, self.dir_name, 'crf.features.tf.per.split.{}.{}'.format(
            suffix, self.prefix), keep_index=True, float_format='%.3f')

    def prepare_train_save(self, f_left, **crf_best_params):
        self.remove_features(f_left)
        self.train(**crf_best_params)
        self.get_report()
        self.save_report(f_left)
        del self.local_doc_map

    def train(self, est=None, **crf_best_params):
        if est:
            self.est = est
        self.crf_res_db, self.crf_f_db = model_utils.prepared_cross_validate_crf(self.dataset,
                                                                                 self.dataset.splits, seq_len=self.seq_len, step=self.step, **crf_best_params)

    def train_save(self, f_left, **crf_best_params):
        self.train(f_left, **crf_best_params)
        self.get_report(f_left)

    def sample_features(self):
        self.sampled = [k.split(
            '.')[0] for k in self.dataset.doc_map[3].sent_list[13].x['1'].keys() if '.' in k]

    def check_features(self, f_left):
        self.sample_features()
        self.f_found = {}
        for left in f_left:
            self.f_found[left] = 0
            if left in self.sampled:
                self.f_found[left] = 1
                break
        for k, v in self.f_found.items():
            if v == 0:
                print('\nfeature {} not found among samples'.format(k), self.sampled)
                print(self.f_found)
                sys.exit()


class MyUndersamplerDoc():
    def __init__(self, doc):
        self.doc = doc

    def undersample(self, ratio=1.0, random_state=42):
        self.ratio = ratio
        self.get_indices(ratio, random_state)
        new_sent_list = []
        for i in self.selected_indices.tolist():
            new_sent_list.append(self.doc.sent_list[i])
        self.doc.sent_list = new_sent_list
        self.save_doc_indices()

    def save_doc_indices(self):
        dir_name = os.path.basename(self.doc.path)
        common_utils.save_json(self.selected_indices.tolist(
        ), dir_name, '{:02d}_selected_indices'.format(self.doc.doc_idx))

    def get_indices(self, ratio=1.0, random_state=42):
        self.reshaped_y = np.array(self.doc.get_y()).reshape(-1, 1)
        self.rus = RandomUnderSampler(
            sampling_strategy=ratio, random_state=random_state)
        _, self.selected_y = self.rus.fit_resample(
            self.reshaped_y, self.reshaped_y)
        self.selected_indices = self.rus.sample_indices_
        self.selected_indices.sort()


class DummyDoc():

    def __init__(self):
        self.y = [0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
        self.sent_list = ['is_nar' if i == 1 else 'not_nar' for i in self.y]
        self.path = 'duplicate'
        self.doc_idx = 99

    def get_y(self):
        return self.y


class MyEnsFold():

    def __init__(self, prepared_splits):
        self.splits = prepared_splits
        self.n_splits = len(prepared_splits.keys())

    def yeld_prepared_splits(self, X, y, groups):
        for split, v in self.splits.items():
            train_idx = [idx for idx, j in enumerate(
                groups) if j in v['train']]
            test_idx = [idx for idx, j in enumerate(groups) if j in v['test']]
            yield train_idx, test_idx

    def split(self, X, y=None, groups=None):
        yield from self.yeld_prepared_splits(X, y, groups)

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits


class MyCrfWrapper():

    def __init__(self, dataset, splits,  seq_len=15, step=15):
        self.dataset = dataset
        self.splits = splits
        self.n_splits = len(self.splits.keys())
        self.seq_len = seq_len
        self.step = step
        self.train_splitter = MyCrfFold(self.splits)
        self.X = {}
        self.y = {}
        self.groups = {}
        self.get_X()

    def get_X(self):
        for k, v in self.splits.items():
            idx = int(k)
            self.X[idx], self.y[idx], self.groups[idx], _ = model_utils.get_X_y_by_doc_indices(
                self.dataset, v['train'], self.seq_len, self.step, k)
            self.X[idx+self.n_splits], self.y[idx+self.n_splits], self.groups[idx+self.n_splits], _ = model_utils.get_X_y_by_doc_indices(
                self.dataset, v['test'], self.seq_len, self.step, k)
        self.y_test = self.y[self.n_splits]
        self.X_test = self.X[self.n_splits]


class MyCrfFold(MyEnsFold):

    def __init__(self, prepared_splits):
        super().__init__(prepared_splits)

    def yeld_prepared_splits(self, X, y, groups):
        for split, v in self.splits.items():
            yield int(split), int(split)+self.n_splits

    def split(self, X=None, y=None, groups=None):
        yield from self.yeld_prepared_splits(X, y, groups)

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits


class PredictionTransformer(BaseEstimator, TransformerMixin, MetaEstimatorMixin):
    def __init__(self, clf):
        """Replaces all features with `clf.predict_proba(X)`"""
        self.clf = clf

    def fit(self, X, y):
        self.clf.fit(X, y)
        return self

    def transform(self, X):
        return self.clf.predict_proba(X)


class ThresholdClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, threshold=0.5):
        """Classify samples based on whether they are above of below `threshold`"""
        self.threshold = threshold
        # print('init >> {} threshold for positive class: {}'.format(
        #     self.__class__.__name__, self.threshold))

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        print(self.classes_)
        return self

    def predict(self, X):
        # the implementation used here breaks ties differently
        # from the one used in RFs:
        # return self.classes_.take(np.argmax(X, axis=1), axis=0)
        print("X shape", X.shape)
        return np.where(X[:, 0] > self.threshold, *self.classes_)


class MyProbWrapper():

    def __init__(self, data, splits_train, splits_test, feature_cols=['bert_proba_0', 'bert_proba_1', 'crf_proba_1', 'crf_proba_1']):
        self.data = data
        self.feature_cols = feature_cols
        self.splits_test = splits_test
        self.splits_train = splits_train
        self.model_prefix = self.feature_cols[0].split('_')[0]
        self.train_splitter = MyEnsFold(self.splits_train)
        self.test_splitter = MyEnsFold(self.splits_test)
        print('init >> {}, model prefix {}, total train splits {}'.format(
            self.__class__.__name__, self.model_prefix, len(self.splits_train)))
        self.get_X()

    def get_X(self):
        self.X = (self.data[self.feature_cols]).to_numpy()
        self.y = (self.data['{}_true'.format(self.model_prefix)]).to_numpy()
        self.groups = (
            self.data['{}_group'.format(self.model_prefix)]).to_numpy()
        self.get_X_test()
        print('group:', self.groups.shape, 'y',
              self.y.shape, 'X', self.X.shape)

    def get_X_test(self):
        for i, (tr, ts) in enumerate(self.test_splitter.split(X=self.X, groups=self.groups)):
            self.X_test = itemgetter(*ts)(self.X)
            self.y_test = itemgetter(*ts)(self.y)


class MyGrid():
    def __init__(self, refit_score='nar_recall_score', recall_weight=0.5):
        print('init >> {}'.format(self.__class__.__name__))
        self.sort_scoring = ['f1_macro_score', refit_score]
        self.refit_score = refit_score
        self.sort_by = ['mean_test_{}'.format(i) for i in self.sort_scoring]
        self.scorers = {
            'nar_precision_score': make_scorer(precision_score, average='binary'),
            'nar_recall_score': make_scorer(recall_score, average='binary'),
            #             'accuracy_score': make_scorer(accuracy_score),
            'f1_weighted_score': make_scorer(f1_score, average='weighted'),
            'f1_macro_score': make_scorer(f1_score, average='macro'),
            'custom_score': make_scorer(self.my_score_func, f1_type='macro', recall_weight=recall_weight)
            # 'f1_micro_score': make_scorer(f1_score, average='micro')
        }
        self.columns = ['mean_test_{}'.format(k) for k in self.scorers.keys()]

    def my_score_func(self, y, y_pred, **kwargs):
        recall = recall_score(y, y_pred, average='binary') * \
            kwargs['recall_weight']
        f1 = f1_score(
            y, y_pred, average=kwargs['f1_type'])*(1-kwargs['recall_weight'])
        return recall+f1

    def add_score(self, dict_):
        self.scorers.update(dict_)
        name = list(dict_.keys())[0]
        self.columns.append('mean_test_{}'.format(name))

    def search_and_shows(self, clf, prob_wrapper,  param_grid, random_search=False):
        self.grid_search_wrapper(clf, prob_wrapper,  param_grid, random_search)
        self.show_grid_results()
        # self.plot('param_n_estimators', self.scorers) # TODO define what param to pass

    def show_grid_results(self):
        self.results = pd.DataFrame(self.grid_search.cv_results_)
        self.results = self.results.sort_values(
            by=self.sort_by, ascending=False)
        display(self.results[self.columns].head())

    def grid_search_wrapper(self, clf, prob_wrapper,  param_grid, random_search=False):
        """
        fits a GridSearchCV classifier using refit_score for optimization
        prints classifier performance metrics
        """
        skf = prob_wrapper.train_splitter  # StratifiedKFold(n_splits=10)
        if random_search:
            self.grid_search = RandomizedSearchCV(clf, param_grid, scoring=self.scorers, refit=self.refit_score,
                                                  cv=skf, return_train_score=True, n_jobs=-1, verbose=4)
        else:
            self.grid_search = GridSearchCV(clf, param_grid, scoring=self.scorers, refit=self.refit_score,
                                            cv=skf, return_train_score=True, n_jobs=-1, verbose=4)
        self.grid_search.fit(
            X=prob_wrapper.X, y=prob_wrapper.y, groups=prob_wrapper.groups)

        # make the predictions
        y_pred = self.grid_search.predict(prob_wrapper.X_test)

        print('Best params for {}'.format(self.refit_score))
        print(self.grid_search.best_params_)

        # confusion matrix on the test data.
        print('\nConfusion matrix of {} optimized for {} on the {} test data:'.format(clf.__class__.__name__,
                                                                                      self.refit_score, len(y_pred)))
        print(pd.DataFrame(confusion_matrix(prob_wrapper.y_test, y_pred),
                           columns=['pred_neg', 'pred_pos'], index=['neg', 'pos']))
        return self.grid_search

    def plot(self, param, scores={}):
        if len(scores) == 0:
            scoring = self.scorers
        else:
            scoring = scores
        scoring_keys = list(scoring.keys())

        results = self.grid_search.cv_results_
        plt.figure(figsize=(13, 13))
        plt.title(
            "GridSearchCV evaluating using {} scorers simultaneously optimized for {}".format(len(scoring), self.refit_score),  fontsize=16)

        plt.xlabel(param.split('_')[-1].upper())
        plt.ylabel("Score")

        ax = plt.gca()
        # ax.set_xlim(0, 402)
        # ax.set_ylim(0.73, 1)

        # Get the regular numpy array from the MaskedArray
        X_axis = np.array(
            results["param_{}".format(param)].data, dtype=float)

        name = "tab10"
        cmap = get_cmap(name)  # type: matplotlib.colors.ListedColormap
        colors = cmap.colors
        total_samples_sum = {}
        total_samples_sum['test'] = np.zeros(
            results['mean_test_{}'.format(scoring_keys[0])].shape[0])
        total_samples_sum['train'] = np.zeros(
            results['mean_train_{}'.format(scoring_keys[0])].shape[0])

        for scorer, color in zip(sorted(scoring), colors):  # ["g", "k"]):
            for sample, style in (("train", "--"), ("test", "-")):
                sample_score_mean = results["mean_%s_%s" % (sample, scorer)]
                sample_score_std = results["std_%s_%s" % (sample, scorer)]
                ax.fill_between(
                    X_axis,
                    sample_score_mean - sample_score_std,
                    sample_score_mean + sample_score_std,
                    alpha=0.1 if sample == "test" else 0,
                    color=color,
                )
                ax.plot(
                    X_axis,
                    sample_score_mean,
                    style,
                    color=color,
                    alpha=1 if sample == "test" else 0.7,
                    label="%s (%s)" % (scorer, sample),
                )
                total_samples_sum[sample] += sample_score_mean

            best_index = np.nonzero(
                results["rank_test_%s" % scorer] == 1)[0][0]
            best_score = results["mean_test_%s" % scorer][best_index]

            # Plot a dotted vertical line at the best score for that scorer marked by x
            ax.plot(
                [
                    X_axis[best_index],
                ]
                * 2,
                [0, best_score],
                linestyle="-.",
                color=color,
                marker="x",
                markeredgewidth=3,
                ms=8,
            )

            # Annotate the best score for that scorer
            ax.annotate("%0.2f" % best_score,
                        (X_axis[best_index], best_score + 0.005))

        best_common_index = np.argmax(total_samples_sum['test']/len(scoring))
        best_common_score = results["mean_test_%s" %
                                    self.refit_score][best_common_index]
        # Plot a dotted vertical line at the best score for all scorers marked by x
        ax.plot(
            [
                X_axis[best_common_index],
            ]
            * 2,
            [0, best_common_score],
            linestyle="solid",
            color=colors[-1],
            marker="x",
            markeredgewidth=3,
            ms=8,
        )

        # Annotate the best score for that scorer
        ax.annotate("%0.2f" % best_score,
                    (X_axis[best_index], best_score + 0.005))

        ax.annotate("%0.2f" % best_common_score,
                    (X_axis[best_common_index], best_common_score + 0.005))
        plt.legend(loc="best")
        # plt.grid(False)
        plt.show()


class MyBooleanRecall():

    def __init__(self, dir_name, pred_db, prefix='ens'):
        self.pred_db = pred_db.copy()
        self.pred_db['abs_idx'] = self.pred_db.index
        self.prefix = prefix
        self.dir_name = dir_name
        self.stat = {}
        print('init >> {} {}'.format(self.__class__.__name__, self.prefix))
        self.assign_true_nar_idx()
        self.get_pred_hit_len()
        self.find_pred_narrative()
        self.get_stat()

    def assign_true_nar_idx(self):
        self.sent_db = common_utils.concat_dbs(
            self.dir_name, "sent_db", ['is_nar', 'nar_idx'], 'sent_idx')
        self.nar_idx_db = self.sent_db.merge(self.pred_db[['{}_true'.format(self.prefix), '{}_predicted'.format(self.prefix), '{}_group'.format(
            self.prefix), 'sent_idx']], left_on=['doc_idx', 'sent_idx'], right_on=['{}_group'.format(self.prefix), 'sent_idx'], validate='one_to_one')
        self.true_nar_len_db = self.nar_idx_db[self.nar_idx_db['is_nar'] == 1].groupby(
            ['doc_idx', 'nar_idx']).size().reset_index(name='nar_len')
        self.stat['total_true_narrative'] = self.true_nar_len_db.shape[0]

    def get_pred_hit_len(self):
        self.pred_hit = self.nar_idx_db[(self.nar_idx_db['{}_true'.format(self.prefix)].eq(self.nar_idx_db['{}_predicted'.format(
            self.prefix)])) & (self.nar_idx_db['{}_true'.format(self.prefix)] == 1)].groupby(['doc_idx', 'nar_idx']).size().reset_index(name='pred_hit_len')
        self.compare = self.true_nar_len_db.merge(self.pred_hit, left_on=['doc_idx', 'nar_idx'], right_on=[
            'doc_idx', 'nar_idx'], how='outer', validate='one_to_one')
        self.compare['hit_ratio'] = self.compare['pred_hit_len'] / \
            self.compare['nar_len']
        self.stat['tp'] = self.compare[~self.compare['pred_hit_len'].isna()
                                       ].shape[0]
        self.stat['fn'] = self.compare[self.compare['pred_hit_len'].isna()
                                       ].shape[0]

    def find_pred_narrative(self):
        cols_to_aggr = ['abs_idx', '{}_predicted'.format(
            self.prefix), '{}_group'.format(self.prefix), '{}_true'.format(self.prefix)]
        self.consecutives_pred = self.pred_db.groupby('{}_group'.format(self.prefix))[
            '{}_predicted'.format(self.prefix)].diff().ne(0).cumsum()
        self.agg_pred = self.pred_db[cols_to_aggr].groupby(
            self.consecutives_pred).agg(list).copy()
        self.agg_pred['is_nar'] = self.agg_pred['{}_predicted'.format(self.prefix)].apply(
            common_utils.get_single_unique)
        self.agg_pred['doc_idx'] = self.agg_pred['{}_group'.format(self.prefix)].apply(
            common_utils.get_single_unique)
        self.agg_pred['hit_true'] = self.agg_pred['{}_true'.format(self.prefix)].apply(
            common_utils.get_single_unique)
        self.agg_pred['nar_len'] = self.agg_pred.abs_idx.map(len)
        self.agg_pred.drop(columns=cols_to_aggr, inplace=True)

    def get_stat(self):
        self.stat['total_pred_narrative'] = self.agg_pred[(self.agg_pred['nar_len'] > 1) & (
            self.agg_pred['is_nar'] == 1)].shape[0]  # how much narratives were predicted in total
        # how much extra narratives were predicted in total
        self.stat['fp'] = self.stat['total_pred_narrative']-self.stat['tp']
        self.stat['recall'] = self.stat['tp'] / \
            (self.stat['tp'] + self.stat['fn'])
        self.stat['precision'] = self.stat['tp'] / \
            (self.stat['tp'] + self.stat['fp'])
        self.stat['f1'] = 2 * (self.stat['recall'] * self.stat['precision']) / \
            (self.stat['recall'] + self.stat['precision'])
        self.stat_db = pd.DataFrame(self.stat, index=[0])


class MyErrorAnalyzer():

    def __init__(self, dir_name, pred_db):
        self.pred_db = pred_db.copy()
        self.dir_name = dir_name
        self.stat = {}
        self.error_db = pd.DataFrame()
        self.error_db['doc_idx'] = pred_db['bert_group'].copy()
        self.error_db['sent_idx'] = pred_db['sent_idx'].copy()
        print('init >> {} '.format(self.__class__.__name__))
        self.get_uniq_error()

    def get_uniq_error(self):
        self.error_db = self.error_db.assign(
            fp_bert=((self.pred_db['bert_predicted'] == 1) & (self.pred_db['bert_true'] == 0)))
        self.error_db = self.error_db.assign(
            fn_bert=((self.pred_db['bert_predicted'] == 0) & (self.pred_db['bert_true'] == 1)))
        self.error_db = self.error_db.assign(
            fp_crf=((self.pred_db['crf_predicted'] == 1) & (self.pred_db['crf_true'] == 0)))
        self.error_db = self.error_db.assign(
            fn_crf=((self.pred_db['crf_predicted'] == 0) & (self.pred_db['crf_true'] == 1)))
        self.error_db = self.error_db.assign(
            fn_ens=((self.pred_db['ens_predicted'] == 0) & (self.pred_db['ens_true'] == 1)))
        self.error_db = self.error_db.assign(
            fp_ens=((self.pred_db['ens_predicted'] == 1) & (self.pred_db['ens_true'] == 0)))
        self.error_db['fp_owner'] = self.error_db.apply(
            lambda x: self.get_who_unique(x.fp_bert, x.fp_crf), axis=1)
        self.error_db['fn_owner'] = self.error_db.apply(
            lambda x: self.get_who_unique(x.fn_bert, x.fn_crf), axis=1)
        self.error_db['ens_fp_action'] = self.error_db.apply(
            lambda x: self.get_ens_action(x.fp_owner, x.fp_ens), axis=1)
        self.error_db['ens_fn_action'] = self.error_db.apply(
            lambda x: self.get_ens_action(x.fn_owner, x.fn_ens), axis=1)
        self.error_db['ens_action'] = self.error_db.apply(
            lambda x: self.get_ens_global_action(x.ens_fp_action, x.ens_fn_action), axis=1)

    def get_ens_global_action(self, x_fp, x_fn):
        for t in ['FIXED', 'KEEP', 'NEW']:
            if x_fp:
                if t in x_fp:
                    return t
            if x_fn:
                if t in x_fn:
                    return t

    def get_ens_action(self, error_owner, ens_error):
        if error_owner:
            if ens_error == False:
                return 'FIXED_{}_ERROR'.format(error_owner)
            else:
                return 'KEEP_{}_ERROR'.format(error_owner)
        elif ens_error == True:
            return 'NEW_ERROR'

    def get_error_type(self, true, pred):
        if true == 1 and pred == 0:
            return 'FN'
        if true == 0 and pred == 1:
            return 'FP'
        # if true == 1 and pred == 1:
        #     return 'TP'
        # if true == 0 and pred == 0:
        #     return 'TN'

    def get_who_unique(self, bert, crf):
        if bert == True and crf == False:
            return 'BERT'
        if bert == False and crf == True:
            return 'CRF'
        if bert == True and crf == True:
            return 'BOTH'

    def plot_fp(self, fp_fn_error_db, hue, plot_cols=[]):
        cols = fp_fn_error_db.columns.tolist()
        plot_cols = [c for c in cols if not any(i in c for i in [
                                                'owner', 'uniq', 'doc', 'idx', 'predicted', 'correct', 'true', 'text', 'type'])]
        plot_cols.sort()
        for col in plot_cols:
            # col='POSTAG_TTL'
            if fp_fn_error_db[col].nunique() < 3:
                type_plot = 'bar'
                (fp_fn_error_db[col].value_counts(
                    normalize=True, sort=False)*100).plot.bar(hue=hue)
                # sns_plot = sns.barplot(data=fp_fn_error_db,
                #                        x=col,
                #                        hue='fn_owner',
                #                        estimator=percentage
                #                        )
                # sns_plot.set(y_label="Percent")
        #         sns_plot =  sns.catplot(data=fp_fn_error_db,
        #                     x=col,
        #                     hue='fn_owner',
        #                    )
                # sns.catplot(
                #     data=fp_fn_error_db, y=col, hue="fn_owner", kind="count",
                #     palette="pastel", edgecolor=".6",
                # )
            else:
                type_plot = 'kde'
                sns_plot = sns.displot(data=fp_fn_error_db,
                                       x=col,
                                       kind=type_plot,
                                       hue=hue,
                                       common_norm=False
                                       )
                sns_plot.set(
                    xlim=(fp_fn_error_db[col].min(), fp_fn_error_db[col].max()))
            sns_plot.set(title=col)
