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
from sklearn.metrics import make_scorer,accuracy_score,balanced_accuracy_score,precision_score,recall_score,f1_score
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate

sys.path.append('./src/')


class TfParams:

    def __init__(self, dir_name, tf_type, stop_list=[], doc_indices=[]):
        self.tf_type = tf_type
        self.stop_list = stop_list
        self.dir_name = dir_name
        self.suffix = '_no.stop' if len(
            stop_list) == 0 else '_stop{}'.format(len(stop_list))
        self.tf = None
        self.doc_indices = doc_indices
        self.set_tf_params()

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

    def build_dict(self):
        self.tf = feature_utils.tfidf_build_all_save_per_doc(
            self.dir_name, self.per_word, self.per_lemma, self.analyzer, self.suffix, self.stop_list, self.doc_indices)
        self.features = self.tf.get_feature_names_out()


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

    def set_features(self, feature_dict=None):
        if type(feature_dict) != dict:
            print("ERROR: not a dict")
        else:
            self.x = feature_dict

    def set_y(self, y=None):
        self.y = y

    def get_x(self):
        return self.x

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
                 merged_str="merged_db",
                 neighbor_radius=3):
        self.sent_list = []
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

    def load_doc_features(self):
        self.doc_db['merged'] = pd.read_csv(os.path.join(
            self.path, "{:02d}_{}.csv".format(self.doc_idx, self.merged_str)))
        self.doc_db['sent_db'] = pd.read_csv(os.path.join(
            self.path, "{:02d}_sent_db.csv".format(self.doc_idx)), usecols=['text', 'par_type', 'nar_idx'])
        self.doc_db['sim_vec'] = pd.read_csv(os.path.join(
            self.path, "{:02d}_sent_sim_vec300_db.csv".format(self.doc_idx)))
        for tf_key, tf_item in self.tf_params.items():
            self.doc_db['tfidf_{}'.format(tf_item.tf_type)] = sparse.load_npz(os.path.join(
                self.path, "{:02d}_tfidf_{}{}.npz".format(self.doc_idx, tf_item.tf_type, tf_item.suffix)))
        self.doc_len = self.doc_db['merged'].shape[0]
        feature_utils.curr_doc_db = self.doc_db

    def pack_doc_features(self):
        for sent_idx in range(self.doc_len):
            par_idx = self.doc_db['merged'].loc[sent_idx, 'par_idx_in_doc']
            sent = Sentence(self.doc_idx, self.doc_len, par_idx,
                            sent_idx, self.doc_db['sent_db'].loc[sent_idx, 'text'],
                            self.doc_db['sent_db'].loc[sent_idx, 'par_type'],
                            self.doc_db['sent_db'].loc[sent_idx, 'nar_idx'])
            sent.set_features(feature_utils.sent2features(
                sent_idx, sent_idx, self.doc_len, self.neighbor_radius))
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

    def pack_doc(self):
        self.load_doc_features()
        self.pack_doc_features()
        self.remove_dbs()

    def reshape_doc(self, seq_len, step):
        shape_name = "{}_{}".format(seq_len, step)
        self.reshaped[shape_name] = [self.sent_list[i: i+seq_len]
                                     for i in np.arange(0, len(self.sent_list), step)]
        print("Doc {} reshaped from {} to {}".format(
            self.doc_idx, self.doc_len, len(self.reshaped[shape_name])))

    def get_x(self, reshaped_name=''):
        x = []
        if not reshaped_name:
            x = [sent.get_x() for sent in self.sent_list]
        else:
            for sent_seq in self.reshaped[reshaped_name]:
                x.append([sent_seq[i].get_x() for i in range(len(sent_seq))])
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
                text.append([sent_seq[i].get_text() for i in range(len(sent_seq))])
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
        self.colored_ind_df['true'] = model_utils.get_colored_from_list(
            self.get_y(), "is_nar")
        self.colored_ind_df['pred'] = model_utils.get_colored_from_list(
            self.get_pred_y(), "is_nar")
        self.write_html("colored_indices")
        return self.print_df

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
                 neighbor_radius=3,
                 doc_indices=[]
                 ):
        self.doc_map = {}
        self.dir_name = dir_name
        self.path = os.path.join(os.getcwd(), defines.PATH_TO_DFS, dir_name)
        self.neighbor_radius = neighbor_radius
        self.merged_str = merged_str
        if len(doc_indices) == 0:
            self.doc_indices = np.arange(1, 81)
        else:
            self.doc_indices = doc_indices
        self.tf_params = {}
        self.nar_df = pd.DataFrame()
        self.print_df = pd.DataFrame()
        print("{} init called".format(self.__class__.__name__))

    def create_tfidf(self):
        for key, tf in self.tf_params.items():
            tf.build_dict()
            print("TdIdf {} built".format(key))

    def pack_dataset(self):
        self.create_tfidf()
        print("\nPacking dataset...")
        for idx in self.doc_indices:
            doc = Document(idx=idx,
                           path=self.path,
                           tf_params=self.tf_params,
                           merged_str=self.merged_str,
                           neighbor_radius=self.neighbor_radius)
            doc.pack_doc()
            print("{}".format(idx, end=' '))
            self.doc_map[idx] = doc

    def reshape(self, seq_len, step):
        print("\nReshaping dataset: seq_len {}, step {}...".format(seq_len, step))
        for idx in self.doc_indices:
            self.doc_map[idx].reshape_doc(seq_len, step)
            print("{}".format(idx, end=' '))

    def get_x(self, doc_indices, reshape_name=''):
        x = []
        for idx in doc_indices:
            x.extend(self.doc_map[idx].get_x(reshape_name))
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
        pickle.dump(self, open(path, "wb"))

    def set_tf_params(self, tf_name, stop_list=[]):
        self.tf_params[tf_name] = TfParams(
            self.dir_name, tf_name, stop_list, self.doc_indices)

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

    def convert_labels(self, y_true, y_pred):
        if isinstance(y_true[0], str):
            y_true = common_utils.convert_str_label_to_binary(y_true)
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


class MyReport():

    def __init__(self):
        print("{} init called".format(self.__class__.__name__))

    def get_avg_scores(self, scr_dict, labels=['not_nar', 'nar']):
        if not isinstance(scr_dict, dict):
            raise Exception("Expect to get dictionary as input")

        avg = {}

        for l in labels:
            avg[l] = {}
        avg['weighted avg'] = {}
        for label in avg.keys():
            avg[label] = {}
            avg[label]['recall'] = []
            avg[label]['prec'] = []
            avg[label]['f1'] = []

            for key, val in scr_dict.items():
                avg[label]['f1'].append(val[label]['f1-score'])
                avg[label]['recall'].append(val[label]['recall'])
                avg[label]['prec'].append(val[label]['precision'])

        for label, item in avg.items():
            avg[label]['avg'] = {}
            for k, v in item.items():
                if k != 'avg':
                    avg[label]['avg'][k] = np.mean(v)
        return avg


class MyScorer():

    def __init__(self):
        print("{} init called".format(self.__class__.__name__))
        self.custom_scorer = {'accuracy': make_scorer(accuracy_score),
                              'balanced_accuracy': make_scorer(balanced_accuracy_score),
                              'precision': make_scorer(precision_score, average='weighted'),
                              'recall': make_scorer(recall_score, average='weighted'),
                              'f1': make_scorer(f1_score, average='weighted'),
                              }
        self.scorer_names=list(self.custom_scorer.keys())
        self.scores_df=pd.DataFrame()

  
    def add_score(self,scores, regressorName, prefix):
        for name in self.scorer_names:
            self.scores_df.loc[regressorName + '_' + prefix, name] = scores["test_"+name].mean()

    
    def get_cross_val_score(self,estimator,X_train,y_train,groups,prefix="",cv=10):
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
            n_jobs = -1
        )
        self.add_score(full_scores,name,prefix)
        end_time = time.time()-start_time
        print("{} took  {}".format(name, time.strftime("%H:%M:%S", time.gmtime(end_time))))
 
    
    
