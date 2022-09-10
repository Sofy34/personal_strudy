import feature_utils
import common_utils
import sys
import pandas as pd
import os
from scipy import sparse
import defines
import numpy as np

sys.path.append('./src/')


class TfParams:

    def __init__(self, dir_name, tf_type, stop_list=[]):
        self.tf_type = tf_type
        self.stop_list = stop_list
        self.dir_name = dir_name
        self.suffix = '_no.stop' if len(
            stop_list) == 0 else '_stop{}'.format(len(stop_list))
        self.tf = None
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
            self.dir_name, self.per_word, self.per_lemma, self.analyzer, self.suffix, self.stop_list)
        self.features = self.tf.get_feature_names_out()


class Sentence:
    def __init__(self,
                 doc_idx,
                 doc_len,
                 par_idx,
                 sent_idx):
        self.doc_idx = doc_idx
        self.sent_idx = sent_idx
        self.doc_len = doc_len
        self.par_idx = par_idx
        self.x = {}
        self.y = []

    def set_features(self, feature_dict=None):
        if type(feature_dict) != dict:
            print("ERROR: not a dict")
        else:
            self.x = feature_dict

    def set_y(self, y=None):
        self.y = y

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y


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

    def load_doc_features(self):
        self.doc_db['merged'] = pd.read_csv(os.path.join(
            self.path, "{:02d}_{}.csv".format(self.doc_idx, self.merged_str)))
        self.doc_db['sim_vec'] = pd.read_csv(os.path.join(
            self.path, "{:02d}_sent_sim_vec300_db.csv".format(self.doc_idx)))
        for tf_key, tf_item in self.tf_params.items():
            self.doc_db['tfidf_{}'.format(tf_item.tf_type)] = sparse.load_npz(os.path.join(
                self.path, "{:02d}_tfidf_{}{}.npz".format(self.doc_idx, tf_item.tf_type, tf_item.suffix)))
        self.doc_len = self.doc_db['merged'].shape[0]
        feature_utils.curr_doc_db = self.doc_db

    def pack_doc_features(self):
        self.load_doc_features()
        for sent_idx in range(self.doc_len):
            par_idx = self.doc_db['merged'].loc[sent_idx, 'par_idx_in_doc']
            sent = Sentence(self.doc_idx, self.doc_len, par_idx, sent_idx)
            sent.set_features(feature_utils.sent2features(
                sent_idx, sent_idx, self.doc_len, self.neighbor_radius))
            sent.set_y(feature_utils.sent2label(sent_idx))
            self.sent_list.append(sent)

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
        for sent_seq in self.reshaped[reshaped_name]:
            x.append([sent_seq[i].get_x() for i in range(len(sent_seq))])
        return x

    def get_y(self, reshaped_name=''):
        y = []
        for sent_seq in self.reshaped[reshaped_name]:
            y.append([sent_seq[i].get_y() for i in range(len(sent_seq))])
        return y

    def get_group(self, reshaped_name=''):
        return [self.doc_idx for seq in self.reshaped[reshaped_name]]


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

    def get_x(self, doc_indices, reshape_name):
        x = []
        for idx in doc_indices:
            x.extend(self.doc_map[idx].get_x(reshape_name))
        return x

    def get_y(self, doc_indices, reshape_name):
        y = []
        for idx in doc_indices:
            y.extend(self.doc_map[idx].get_y(reshape_name))
        return y

    def get_group(self, doc_indices, reshape_name):
        group = []
        for idx in doc_indices:
            group.extend(self.doc_map[idx].get_group(reshape_name))
        return group

    def save_to_json(self, file_name):
        common_utils.save_json(self.doc_map, self.dir_name, file_name, False)

    def set_tf_params(self, tf_name, stop_list=[]):
        self.tf_params[tf_name] = TfParams(self.dir_name, tf_name, stop_list)
