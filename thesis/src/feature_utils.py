import numpy as np
import pandas as pd
import defines

from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, PassiveAggressiveClassifier, Perceptron, RidgeClassifier, RidgeClassifierCV, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer , TfidfVectorizer, TfidfTransformer
from sklearn.preprocessing import normalize, StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn_crfsuite import scorers, CRF, metrics
from sklearn.metrics import ConfusionMatrixDisplay #flat_classification_report
from nltk import tokenize
from bidi import algorithm as bidialg      # needed for arabic, hebrew
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt
import fasttext.util
import os
from sklearn.metrics.pairwise import cosine_similarity
import glob
from scipy import sparse
import json
from operator import itemgetter
from sklearn_crfsuite import scorers, CRF
from sklearn_crfsuite import metrics
from sklearn_crfsuite.metrics import flat_classification_report
from sklearn.model_selection import LeaveOneGroupOut,LeavePGroupsOut,GroupKFold

import itertools
import common_utils

regressors_instance = {}
regressors_prediction = {}
regressors_type = [
    LogisticRegression(random_state=0),
    LogisticRegressionCV(random_state=0),
    PassiveAggressiveClassifier(random_state=0),
    Perceptron(random_state=0),
    RidgeClassifier(random_state=0),
    RidgeClassifierCV(),
    SGDClassifier(random_state=0),
    SVC(random_state=0),
    DecisionTreeClassifier(random_state=0)
]

scores_df = pd.DataFrame(dtype=float)
def load_fasstex_model():
    ft = fasttext.load_model('./external_src/cc.he.300.bin')
    return ft


### EMBEDDED VECTORS ###

def get_and_save_sent_vectors(dir_name,doc_idx,ft,dim = 300): 
    sent_db = pd.read_csv(os.path.join(os.getcwd(),defines.PATH_TO_DFS,dir_name,"{:02d}_sent_db.csv".format(doc_idx)))
    sent_vec_db = get_vector_per_sentence(sent_db,ft,dim)
    sent_vec_db.to_csv(os.path.join(os.getcwd(),defines.PATH_TO_DFS,dir_name,"{:02d}_sent_vec{}_db.csv".format(doc_idx,dim)),index=False)
    print("{} doc sent saved".format(doc_idx,dim))

def get_vector_per_sentence(db, ft, dim = 300):
    if (dim < 300):
        fasttext.util.reduce_model(ft, dim)
    sent_vectors = [ft.get_sentence_vector(row['text']) for index, row in db.iterrows()]
    # sent_array = np.vstack(sent_vectors)
    sent_vec_db = pd.DataFrame(sent_vectors)
    return sent_vec_db

def get_and_save_doc_similarity(dir_name,doc_idx,dim = 300): 
    sent_vec_db = pd.read_csv(os.path.join(os.getcwd(),defines.PATH_TO_DFS,dir_name,"{:02d}_sent_vec{}_db.csv".format(doc_idx,dim)))
    sim_db = pd.DataFrame(cosine_similarity(sent_vec_db))
    sim_db.to_csv(os.path.join(os.getcwd(),defines.PATH_TO_DFS,dir_name,"{:02d}_sent_sim_vec{}_db.csv".format(doc_idx,dim)),index=False)
    print("{} sim_db sent saved".format(doc_idx))

#########################

### STOP WORDS RATE ###
def count_stop_words_per_sent(sentence):
    words =  tokenize.word_tokenize(sentence)
    sent_stop_words = [w for w in words if w in stop_words]
    rate = len(sent_stop_words)/len(words)
    return rate

def get_stop_words_rate(doc_idx):
    sent_db = pd.read_csv(os.path.join(os.getcwd(),defines.PATH_TO_DFS,"{:02d}_sent_db.csv".format(doc_idx)))
    for i in sent_db.index:
        sent_db.loc[i,'stop_words_rate'] = count_stop_words_per_sent(i)

#########################

### TF-IDF ###

def get_all_docs_lemma():
    all_docs_lemma = pd.concat(map(pd.read_csv, glob.glob(os.path.join(os.getcwd(),defines.PATH_TO_DFS, "*_sent_lemma_db.csv"))),axis=0)
    all_docs_lemma.reset_index(inplace=True)
    return all_docs_lemma




def get_all_doc_sentenses():
    all_docs_sent = pd.concat(map(pd.read_csv, glob.glob(os.path.join(os.getcwd(),defines.PATH_TO_DFS, "*_sent_db.csv"))),axis=0)
    all_docs_sent.reset_index(inplace=True)
    return all_docs_sent

def tfidf_transform_doc(doc_idx,tfidf,per_lemma=True):
    if per_lemma:
        sent_lemma_db = pd.read_csv(os.path.join(os.getcwd(),defines.PATH_TO_DFS,"{:02d}_sent_lemma_db.csv".format(doc_idx)),usecols=['sent_lemma'])
        corpus = sent_lemma_db['sent_lemma'].tolist()
    else:
        sent_db = pd.read_csv(os.path.join(os.getcwd(),defines.PATH_TO_DFS,"{:02d}_sent_db.csv".format(doc_idx)),usecols=['text'])
        corpus = sent_db['text'].tolist()
    return tfidf.transform(corpus)


def tfidf_fit(per_word = True, per_lemma = True, analyzer = 'char',n_min = 3, n_max = 5, min_df = 5):
    data_list =  []
    if per_word:
        if per_lemma:
            corpus = get_all_docs_lemma()
            col_name = 'sent_lemma'
        else:
            corpus = get_all_doc_sentenses()
            col_name = 'text'
        data_list = corpus[col_name].tolist()
        tfidf = TfidfVectorizer(lowercase=False)
    else:
        corpus = get_all_doc_sentenses()
        data_list = corpus['text'].tolist()
        tfidf = TfidfVectorizer(lowercase=False,
        analyzer = analyzer,
        ngram_range = (n_min,n_max),
        min_df = min_df
        )
    return tfidf.fit(data_list)

def tfidf_build_all_save_per_doc(per_word = True,per_lemma=True,analyzer = 'char'):
    tf = tfidf_fit(per_word,per_lemma,analyzer)
    tf_params = tf.get_params()
    tf_string = tf_params['analyzer']
    print("TfIdf {} vocab size {}".format(tf_string,len(tf.vocabulary_)))
    features = tf.get_feature_names()
    sample_features(features)
    sent_lemma_db_list = glob.glob(os.path.join(os.getcwd(),defines.PATH_TO_DFS, "*_sent_lemma_db.csv"))
    for i,doc_name in enumerate(sent_lemma_db_list):
        doc_prefix = common_utils.get_doc_idx_from_name(doc_name)
        X = tfidf_transform_doc(doc_prefix,tf,per_lemma)
        sparse.save_npz(os.path.join(os.getcwd(),defines.PATH_TO_DFS,"{:02d}_tfidf_{}.npz".format(doc_prefix,tf_string)), X)
        print("TfIdf {} saved".format(doc_prefix))





#########################

## POS from YAP ###

def get_and_save_sent_lemma_db(dir_name,doc_idx):
    doc_name = os.path.join(os.getcwd(),defines.PATH_TO_DFS,dir_name,"{:02d}_sent_pos_db.csv".format(doc_idx))
    if not os.path.isfile(doc_name):
        print("ERROR: {} does not exists".format(doc_name))
        return
    sent_pos_db = pd.read_csv(doc_name,usecols=['sent_idx','LEMMA'])
    sent_lemma_db = pd.DataFrame()
    sent_lemma_db['sent_lemma'] = sent_pos_db.groupby('sent_idx')['LEMMA'].apply(lambda x: "%s" % ' '.join(x)).tolist()
    sent_lemma_db.to_csv(os.path.join(os.getcwd(),defines.PATH_TO_DFS,dir_name,"{:02d}_sent_lemma_db.csv".format(doc_idx)),index=False)
    print("{} sent lemma db saved".format(doc_idx))



def get_and_save_sent_pos_count_db(dir_name,doc_idx):
    columns_to_count = ['POSTAG','f_gen','f_num','f_suf_gen','f_suf_num','f_suf_per','f_per','f_tense']
    sent_pos_db = pd.read_csv(os.path.join(os.getcwd(),defines.PATH_TO_DFS,dir_name,"{:02d}_sent_pos_db.csv".format(doc_idx)))
    sent_pos_dummies = pd.get_dummies(sent_pos_db,columns=columns_to_count)
    sent_pos_dummies.fillna(value=0,inplace=True)
    count_db = sent_pos_dummies.groupby('sent_idx').sum()
    count_db['TOKEN'] = sent_pos_dummies.groupby('sent_idx')['TOKEN'].max()
    count_db.drop(['FROM','TO','doc_idx'],inplace=True,axis=1)
    normalize_pos_count_on_sent_len(count_db)
    count_db.to_csv(os.path.join(os.getcwd(),defines.PATH_TO_DFS,dir_name,"{:02d}_sent_pos_count_db.csv".format(doc_idx)),index=False)
    print("{} sent count db saved".format(doc_idx))


def normalize_pos_count_on_sent_len(count_db):
    count_db.iloc[:,1:] = count_db.iloc[:,1:].div(count_db.TOKEN, axis=0)
#########################

### Merge all sentense features into single DB ###
def merge_sent_pos_db(dir_name,doc_idx):
    sent_db = pd.read_csv(os.path.join(os.getcwd(),defines.PATH_TO_DFS,dir_name,"{:02d}_sent_db.csv".format(doc_idx)),usecols=defines.SENT_FEATURES)
    count_db = pd.read_csv(os.path.join(os.getcwd(),defines.PATH_TO_DFS,dir_name,"{:02d}_sent_pos_count_db.csv".format(doc_idx)))
    merged_db =  pd.merge(sent_db,count_db, left_index=True,right_index=True,validate="one_to_one")
    merged_db.to_csv(os.path.join(os.getcwd(),defines.PATH_TO_DFS,dir_name,"{:02d}_merged_db.csv".format(doc_idx)),index=False)
    print("{} sent features db saved".format(doc_idx))

#########################

### Pack sentense features for CRF  ###

curr_doc_db = {}
def load_doc_features(dir_name,doc_idx,tf_types = ['word','char_wb']):
    global curr_doc_db
    curr_doc_db['merged'] = pd.read_csv(os.path.join(os.getcwd(),defines.PATH_TO_DFS,dir_name,"{:02d}_merged_db.csv".format(doc_idx)))
    curr_doc_db['sim_vec']  = pd.read_csv(os.path.join(os.getcwd(),defines.PATH_TO_DFS,dir_name,"{:02d}_sent_sim_vec300_db.csv".format(doc_idx)))
    for tf_type in tf_types:
        curr_doc_db['tfidf_{}'.format(tf_type)] = sparse.load_npz(os.path.join(os.getcwd(),defines.PATH_TO_DFS,dir_name,"{:02d}_tfidf_{}.npz".format(doc_idx,tf_type)))
    return curr_doc_db

def save_doc_packed_features(doc_idx,dictionary_data):
    j_file = open(os.path.join(os.getcwd(),defines.PATH_TO_DFS,"{:02d}_packed_dict.csv".format(doc_idx)), "w")
    json.dump(dictionary_data, j_file)
    print("{} packed features saved".format(doc_idx))
    j_file.close()



def pack_doc_features(doc_idx,seq_len = 6,step = 6):
    X_doc = [doc2features(first_sent_idx) for first_sent_idx in curr_doc_db['merged'].index[:-seq_len:step]] # TBD take last indices as well
    # save_doc_packed_features(doc_idx,X_doc) #TBD open after solve 'Object of type int64 is not JSON serializable'
    y_doc = [doc2labels(first_sent_idx) for first_sent_idx in curr_doc_db['merged'].index[:-seq_len:step]]
    groups_doc =  [doc_idx for i in range(len(y_doc))]
    return X_doc,y_doc,groups_doc

def pack_all_doc_features(seq_len = 6,step = 6):
    X = []
    y = []
    groups = []
    # doc_db_path = os.path.join(os.getcwd(),defines.PATH_TO_DFS,"doc_db.csv")
    # if os.path.isfile(doc_db_path):
    #     doc_db = pd.read_csv(doc_db_path)
    merged_db_list = glob.glob(os.path.join(os.getcwd(),defines.PATH_TO_DFS, "*_merged_db.csv"))
    merged_db_list.sort()
    for doc_name in merged_db_list:
        doc_idx = common_utils.get_doc_idx_from_name(doc_name)
        load_doc_features(doc_idx)
        X_doc,y_doc,groups_doc =  pack_doc_features(doc_idx)
        X.extend(X_doc)
        y.extend(y_doc)
        groups.extend(groups_doc)
        
    print("Features packed for {} docs".format(len(doc_db.index)))
    return X,y,groups

def pack_doc_sentences(doc_idx):
    doc_len =  len(curr_doc_db['merged'].index)
    X_doc = [sent2features(sent_idx,sent_idx,doc_len) for sent_idx in curr_doc_db['merged'].index] 
    # save_doc_packed_features(doc_idx,X_doc) #TBD open after solve 'Object of type int64 is not JSON serializable'
    y_doc = [sent2label(sent_idx) for sent_idx in curr_doc_db['merged'].index]
    groups_doc =  [doc_idx for i in range(len(y_doc))]
    print ("{} doc {} sentences packed".format(doc_idx,doc_len))
    return X_doc,y_doc,groups_doc

def reshape_doc_features_to_sequence(X,y,doc_idx,seq_len,step):
    X_seq = [X[i : i+seq_len] for i in np.arange(0,len(X),step)]
    y_seq = [y[i : i+seq_len] for i in np.arange(0,len(y),step)]
    groups_seq = [doc_idx for i in range(len(y_seq))]
    print ("doc sentences reshaped: from {} to {}".format(len(X),len(X_seq)))
    return X_seq,y_seq,groups_seq


def reshape_doc_paragraphs_to_sequence(X,y,doc_idx,seq_len,step):
    X_seq =  [list(itertools.chain.from_iterable(X[i : i+seq_len])) for i in np.arange(0,len(X),step)]
    y_seq = [list(itertools.chain.from_iterable(y[i : i+seq_len])) for i in np.arange(0,len(y),step)]
    groups_seq = [doc_idx for i in range(len(y_seq))]
    print ("doc paragraphs reshaped: from {} to {}".format(len(X),len(X_seq)))
    return X_seq,y_seq,groups_seq

def reshape_doc_paragraphs_to_sequence_by_len(X,y,groups,seq_len,step):
    X_seq =  [list(itertools.chain.from_iterable(X[i : i+seq_len])) for i in np.arange(0,len(X),step)]
    y_seq = [list(itertools.chain.from_iterable(y[i : i+seq_len])) for i in np.arange(0,len(y),step)]
    groups_seq = groups[::step]
    print ("doc paragraphs reshaped: from {} to {}".format(len(X),len(X_seq)))
    return X_seq,y_seq,groups_seq

def pack_doc_per_paragraph(doc_idx,doc_as_sequence):
    par_indices = curr_doc_db['merged']['par_idx_in_doc'].unique()
    # print("Doc {} had {} paragraphs".format(doc_idx,len(par_indices)))
    X_doc = [par2features(par_idx,doc_as_sequence) for par_idx in par_indices] 
    # save_doc_packed_features(doc_idx,X_doc) #TBD open after solve 'Object of type int64 is not JSON serializable'
    y_doc = [par2label(par_idx) for par_idx in par_indices]
    groups_doc =  [doc_idx for i in range(len(y_doc))]
    print ("{} doc {} paragraphes packed".format(doc_idx,len(par_indices)))
    return X_doc,y_doc,groups_doc

def par2features(par_idx,doc_as_sequence): #doc_len is dummy variable
    par_len = curr_doc_db['merged'].query("par_idx_in_doc == @par_idx").shape[0]
    par_sentences = curr_doc_db['merged'].query("par_idx_in_doc == @par_idx").index
    doc_sent_num = len(curr_doc_db['merged'].index)
    # print("par {} has {} sentenses".format(par_idx,par_len))
    if doc_as_sequence:
        X_par = [sent2features(sent_idx,sent_idx,doc_sent_num) for sent_idx in par_sentences] 
    else:
        X_par = [sent2features(sent_idx,idx_in_seq,par_len) for idx_in_seq,sent_idx in enumerate(par_sentences)] 
    return X_par

def pack_doc_per_paragraph_limit(doc_idx,limit,doc_as_sequence):# 0 means no limit
    par_indices = curr_doc_db['merged']['par_idx_in_doc'].unique()
    X_doc = []
    y_doc = []
    groups_doc = []
    for par_idx in par_indices:
        sub_par_x_list = par2features_limit(par_idx,limit,doc_as_sequence)
        sub_par_y_list = par2label_limit(par_idx,limit)
        for i in range(len(sub_par_x_list)):
            X_doc.append(sub_par_x_list[i])
            y_doc.append(sub_par_y_list[i])
    groups_doc =  [doc_idx for i in range(len(y_doc))]
    # print ("Doc {} has {} paragraphes, packed by limit {} = {} paragraphs".format(doc_idx,len(par_indices),limit,len(y_doc)))
    return X_doc,y_doc,groups_doc

def reshape_docs_map_to_seq(docs_map,per_par,seq_len,step):
    for doc in docs_map.keys():
        if per_par:
            X_doc,y_doc,groups_doc = reshape_doc_paragraphs_to_sequence(docs_map[doc]['X'],docs_map[doc]['y'],doc,seq_len,step)
        else:
            X_doc,y_doc,groups_doc = reshape_doc_features_to_sequence(docs_map[doc]['X'],docs_map[doc]['y'],doc,seq_len,step)
        docs_map[doc]['X_{}_{}'.format(seq_len,step)] = X_doc
        docs_map[doc]['y_{}_{}'.format(seq_len,step)] = y_doc
        docs_map[doc]['groups_{}_{}'.format(seq_len,step)] = groups_doc
    # return docs_map

def par2label_limit(par_idx,limit=0):
    par_len = curr_doc_db['merged'].query("par_idx_in_doc == @par_idx").shape[0]
    sent_indices = curr_doc_db['merged'].query("par_idx_in_doc == @par_idx").index
    y_par = []
    for i in np.arange(0,par_len,limit):
        sub_indices = sent_indices[i:min(par_len,i+limit)]
        y_par.append([sent2label(sent_idx) for sent_idx in sub_indices])
    return y_par

def par2features_limit(par_idx,limit,doc_as_sequence): #limit maximum sentences per paragraph
    par_len = curr_doc_db['merged'].query("par_idx_in_doc == @par_idx").shape[0]
    sent_indices = curr_doc_db['merged'].query("par_idx_in_doc == @par_idx").index
    doc_sent_num = len(curr_doc_db['merged'].index)
    X_par = []
    for i in np.arange(0,par_len,limit):#0-7, 8-15....
        sub_indices = sent_indices[i:min(par_len,i+limit)]
        if doc_as_sequence:
            sub_par = [sent2features(sent_idx,sent_idx,doc_sent_num) for sent_idx in sub_indices] 
        else:
            sub_par = [sent2features(sent_idx,idx,len(sub_indices)) for idx,sent_idx in enumerate(sub_indices)] 
        X_par.append(sub_par)
    return X_par




def doc_sent2features_db(dox_idx):
    load_doc_features(doc_idx,defines.TF_TYPES)
    doc_features_db = pd.DataGrame()
    sent_indices = curr_doc_db['merged'].index
    sent_count = len(sent_indices)
    sent_indices = curr_doc_db['merged'].index
    for sent_idx in sent_indices:
        doc_features_db[sent_idx] = sent2features(sent_idx,sent_idx,sent_count)

    
def par2label(par_idx):
    y_par = [sent2label(sent_idx) for sent_idx in curr_doc_db['merged'].query("par_idx_in_doc == @par_idx").index]
    return y_par

def pack_reshape_all_doc_sentences(seq_len,step,per_par = False):
    X = []
    y = []
    groups = []
    doc_db_path = os.path.join(os.getcwd(),defines.PATH_TO_DFS,"doc_db.csv")
    if os.path.isfile(doc_db_path):
        doc_db = pd.read_csv(doc_db_path)
    for doc_idx in doc_db.doc_idx_from_name:
        load_doc_features(int(doc_idx))
        if per_par:
            X_doc,y_doc,groups_doc = pack_doc_per_paragraph(doc_idx)
            X_seq,y_seq,groups_seq = reshape_doc_paragraphs_to_sequence(X_doc,y_doc,doc_idx,seq_len,step)
        else:
            X_doc,y_doc,groups_doc =  pack_doc_sentences(doc_idx)
            X_seq,y_seq,groups_seq = reshape_doc_features_to_sequence(X_doc,y_doc,groups_doc,seq_len,step)
        X.extend(X_seq)
        y.extend(y_seq)
        groups.extend(groups_seq)
        
    print("Sentenced packed for {} docs".format(len(doc_db.index)))
    return X,y,groups

def pack_all_doc_sentences(per_par=False,tf_types = ['word','char_wb']):
    X = []
    y = []
    groups = []
    sent_lemma_db_list = glob.glob(os.path.join(os.getcwd(),defines.PATH_TO_DFS, "*_sent_lemma_db.csv"))
    # doc_db_path = os.path.join(os.getcwd(),defines.PATH_TO_DFS,"doc_db.csv")
    # if os.path.isfile(doc_db_path):
    #     doc_db = pd.read_csv(doc_db_path)
    for doc_name in sent_lemma_db_list:
        doc_idx = common_utils.get_doc_idx_from_name(doc_name) 
        load_doc_features(doc_idx,tf_types)
        if per_par:
            X_doc,y_doc,groups_doc =  pack_doc_per_paragraph(doc_idx)
        else:
            X_doc,y_doc,groups_doc =  pack_doc_sentences(doc_idx)
        X.extend(X_doc)
        y.extend(y_doc)
        groups.extend(groups_doc)
        # print("Doc {} sentenced packed".format(doc_idx))
        
    print("{} sentenced packed for {} docs".format(len(X),len(sent_lemma_db_list)))
    return X,y,groups

def pack_all_doc_sentences_to_map(dir_name,per_par=False,limit=0,doc_as_sequence=0,sent_lemma_db_list=[],tf_types = ['word','char_wb']):
    docs_map = {}
    if len(sent_lemma_db_list) ==0:
        sent_lemma_db_list = glob.glob(os.path.join(os.getcwd(),defines.PATH_TO_DFS, dir_name,"*_sent_lemma_db.csv"))    
    total_sent = 0
    for doc_name in sent_lemma_db_list:
        doc_idx = common_utils.get_doc_idx_from_name(doc_name) 
        load_doc_features(dir_name,doc_idx,tf_types)
        docs_map[doc_idx] = {}
        if per_par:
            if limit == 0:
                X_doc,y_doc,groups_doc =  pack_doc_per_paragraph(doc_idx,doc_as_sequence)
            else:
                X_doc,y_doc,groups_doc = pack_doc_per_paragraph_limit(doc_idx,limit,doc_as_sequence)
        else:
            X_doc,y_doc,groups_doc =  pack_doc_sentences(doc_idx)
        docs_map[doc_idx]['X'] = X_doc
        docs_map[doc_idx]['y']= y_doc
        docs_map[doc_idx]['groups'] = groups_doc
        total_sent += len(docs_map[doc_idx]['X'])
        # print("Doc {} sentenced packed".format(doc_idx))
        
    print("{} items packed for {} docs".format(total_sent,len(docs_map.keys())))
    return docs_map


def sent2features(sent_idx,idx_in_seq,seq_len=6,neighbor_radius =2):
    global curr_doc_db
    features = {}
    columns =  list(curr_doc_db['merged'].columns.values)
    columns.remove('is_nar')
    # print ("Parsing sent idx {} idx_in_seq {} seq_len {}".format(sent_idx,idx_in_seq,seq_len))
    for col in columns:
        if save_feature_value(curr_doc_db['merged'].loc[sent_idx,col],col):
            features["{}".format(col)]= curr_doc_db['merged'].loc[sent_idx,col]

    if idx_in_seq > 1:
        update = {}
        for col in columns:
            if save_feature_value(curr_doc_db['merged'].loc[sent_idx-1,col],col):
                update["-1:{}".format(col)]=curr_doc_db['merged'].loc[sent_idx-1,col]
        features.update(update)

    
    if idx_in_seq > 2:
        update = {}
        for col in columns:
            if save_feature_value(curr_doc_db['merged'].loc[sent_idx-2,col],col):
                update["-2:{}".format(col)]=curr_doc_db['merged'].loc[sent_idx-2,col]
        features.update(update)
    
    update = {}
    for neighbor_dist in range(1,neighbor_radius+1):
        if idx_in_seq > neighbor_dist - 1:
            update["-{}.sim".format(neighbor_dist)]=curr_doc_db['sim_vec'].iloc[sent_idx,sent_idx-neighbor_dist]
        if idx_in_seq < seq_len - neighbor_dist:
            update["+{}.sim".format(neighbor_dist)]=curr_doc_db['sim_vec'].iloc[sent_idx,sent_idx+neighbor_dist]

    features.update(update) 
    
    update = {}
    for tf_type in defines.TF_TYPES:
        tf_str = 'tfidf_{}'.format(tf_type)
        if  tf_str in curr_doc_db.keys():
            tfidf_feature_indices = curr_doc_db[tf_str][sent_idx,:].nonzero()[1]
            for i in tfidf_feature_indices:
                update["{}_{}".format(tf_str,i)] = curr_doc_db[tf_str][sent_idx,i]
            features.update(update)

    if idx_in_seq < seq_len-1:
        update = {}
        for col in columns:
            if save_feature_value(curr_doc_db['merged'].loc[sent_idx+1,col],col):
                update["+1:{}".format(col)]=curr_doc_db['merged'].loc[sent_idx+1,col]
        features.update(update)

    if idx_in_seq < seq_len-2:
        update = {}
        for col in columns:
            if save_feature_value(curr_doc_db['merged'].loc[sent_idx+2,col],col):
                update["+2:{}".format(col)]=curr_doc_db['merged'].loc[sent_idx+2,col]
        features.update(update)

    return features

def save_feature_value(value,name):
    save = 0
    if value !=0 or 'idx' in name or 'is' in name:
        save = 1
    else:
        save = 0
    return save

def sent2label(sent_idx):
    return "is_nar" if curr_doc_db['merged'].loc[sent_idx,'is_nar'].astype(bool)==True else "not_nar"


def doc2features(first_sent_idx,seq_len=6,seq_step=6):
    return [sent2features(sent_idx,idx_in_seq) for idx_in_seq,sent_idx in enumerate(np.arange(first_sent_idx,first_sent_idx+seq_len,dtype=int))]
def doc2labels(first_sent_idx,seq_len=6,seq_step=6):
    return [sent2label(sent_idx) for sent_idx in range(first_sent_idx,first_sent_idx+seq_len)]

#########################

def get_num_text_union(df):
    numeric_cols = df.columns[df.columns.dtype != object].tolist()
    
    transformer_text = FunctionTransformer(lambda x: x['narrative'], validate=False)
    transfomer_numeric = FunctionTransformer(lambda x: x[numeric_cols], validate=False)

    pipeline = Pipeline([
        ('features', FeatureUnion([
                ('numeric_features', Pipeline([
                    ('selector', transfomer_numeric)
                ])),
                 ('text_features', Pipeline([
                    ('selector', transformer_text),
                    (regr_text.__class__.__name__, regr_text)
                ]))
             ])),
        ('estimator', regr_num)
    ])
    
    f_union = FeatureUnion([
                ('numeric_features', Pipeline([
                    ('selector', transfomer_numeric)
                ])),
                 ('text_features', Pipeline([
                    ('selector', transformer_text),
                    (regr_text.__class__.__name__, regr_text)
                ]))
             ])
    
    return pipeline

def process_text_to_features():
    tdidf = TfidfVectorizer(min_df=4,norm='l1')
    tdidf.fit_transform(text_train)



def get_label_and_drop(_df):
    df = _df.copy()
    label = df[defines.LABEL]
    df = drop_columns(df,[defines.LABEL])
    return df, label

def drop_columns(df, columns):
    return df.copy().drop(columns, axis=1)


# machine learning tils
def load_stop_words():
    stop_words = [x.strip() for x in open('heb_stopwords.txt','r').read().split('\n')]
    return stop_words

def plot_important_features(coef, feature_names, top_n=20, ax=None, rotation=60):
    if ax is None:
        ax = plt.gca()
    inds = np.argsort(coef)
    low = inds[:top_n]
    high = inds[-top_n:]
    important = np.hstack([low, high])
    myrange = range(len(important))
    colors = ['red'] * top_n + ['blue'] * top_n
    
    ax.bar(myrange, coef[important], color=colors)
    ax.set_xticks(myrange)
    heb_feature_names =[bidialg.get_display(feature) for feature in feature_names[important]]
    ax.set_xticklabels(heb_feature_names, rotation=rotation, ha="right")
    ax.set_xlim(-.7, 2 * top_n)
    ax.set_frame_on(False)


def get_yap_tag_description(tag):
    desc = defines.YAP_TAG_DICT['MY_POS_TAG_DICT'].get(tag,tag)
    # if desc is not None:
    #     return desc
    # yap_tag = defines.HEB2UDPOS_DICT.get(tag,None)
    # if yap_tag is None:
    #     return tag
    # for key, value  in defines.YAP_TAG_DICT.items():
    #     desc = value.get(yap_tag,'')
    #     if len(desc)!=0 :
    #         return desc
    return desc

def sample_features(features):
    print("Sample of {} features".format(len(features)))
    n = len(features)
    print(features[:20])
    print(features[round(n/2)-10:round(n/2)+10])
    print(features[::round(n/10)])
    
def get_train_test_text(X,y):
    X_train, X_test, y_train, y_test = split_and_get_text(X, y)
    text_train = X_train['text'].to_list()
    text_test = X_test['text'].to_list()
    print ("len train: {}, len test: {}".format(len(text_train),len(text_test)))
    return text_train, text_test, y_train, y_test


def split_db(_db):
    data,label  = get_label_and_drop(_db)
    X_train, X_test, y_train, y_test = train_test_split(data, label, stratify=label, random_state=0)
    return X_train, X_test, y_train, y_test


def run_classifier(_db):
    X_train, X_test, y_train, y_test = split_db(_db)
    sgc = SGDClassifier()
    sgc.fit(X_train, y_train)
    y_pred = sgc.predict(X_test)
    plot_confusion_matrix(sgc, X_test, y_test, cmap='gray_r')
    print(classification_report(y_test, y_pred))    

def run_model(db):
    X,y = get_label_and_drop(db)
    text_train,text_test,y_train, y_test = get_train_test_text(X,y)

    tdif = TfidfVectorizer(stop_words=load_stop_words(),min_df=4)


    X_train_vec = tdif.fit_transform(text_train)
    X_train_vec = normalize(X_train_vec,norm="l1")

    X_test_vec = tdif.transform(text_test)
    X_test_vec = normalize(X_test_vec,norm="l1")

    feature_names = tdif.get_feature_names()
    sample_features(feature_names)

    sgc = SGDClassifier()
    sgc.fit(X_train_vec, y_train)
    y_pred = sgc.predict(X_test_vec)

    plt.figure(figsize=(15, 6))
    plot_important_features(sgc.coef_.ravel(), np.array(tdif.get_feature_names()), top_n=20, rotation=40)
    ax = plt.gca()
    plt.show()

    plot_confusion_matrix(sgc, X_test_vec, y_test, cmap='gray_r')
    print(classification_report(y_test, y_pred))


def get_prediction_report(y_test,y_pred,labels):
    # ConfusionMatrixDisplay.from_predictions(y_test, y_pred,cmap='gray_r')
    cm = confusion_matrix(y_test, y_pred, labels = labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = labels)
    print(classification_report(y_test, y_pred,labels=labels))
    disp.plot(cmap='gray_r')

def split_and_get_text(X,y):
    print ("total data len: {}".format(len(y)))
    X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=101,stratify=y)
    return X_train, X_test, y_train, y_test

def get_random_par(db,is_nar,len_threshold=30):
    return db.query("is_nar==1 & par_len >= @len_threshold").sample(n=1)


def add_features_prev_is_nar(_db):
    # global db = globals()[_db]
    db = _db.copy()
    db['one_before_is_nar']=db['is_nar'].shift(periods=1, fill_value=0)
    db['two_before_is_nar']=db['is_nar'].shift(periods=2, fill_value=0)
    return db
# utilities function are same as were implemented
# with Alexander Kruglyak for assigments during the semester

def show_data_basic_information(df):
    print("Info\n")
    print(df.info())
    print("\n" + "*" * 10 + "\n")
    
    print("Shape\n")
    print(df.shape) 
    print("\n" + "*" * 10 + "\n")
    
    print("Amount of is null data\n")
    print(df.isnull().sum().max())
    print("\n" + "*" * 10 + "\n")
    
    print("Describe\n")
    display(df.describe())
    print("\n" + "*" * 10 + "\n")
    
def drop_columns(df, columns):
    return df.copy().drop(columns, axis=1)


def show_random_text(_df,feature,n=1):
    df = _df.sample(n=n,random_state=42)
    print(list(df[feature]))
    
def get_cross_val_score(estimator,X_train,y_train,prefix="",sampler=None):
        global scores_
        name = estimator.__class__.__name__
        pipe = estimator
        sampler_name = ""
        if sampler is not None:
            pipe = make_imb_pipeline(sampler(random_state=42), estimator)
            sampler_name = sampler.__name__
        print('*********' + name + ' ' + sampler_name + '*********')
        full_scores = cross_validate(
            pipe,
            X_train, 
            y_train, 
            cv=10,
            scoring=('roc_auc', 'average_precision', 'recall', 'f1'),
            n_jobs = -1
        )
        add_score(full_scores, estimator.__class__.__name__,prefix)

def save_estimator(estimator):
    global regressors_instance
    regressors_instance[estimator.__class__.__name__] = estimator

def add_score(scores, regressorName, dataType):
    global scores_df
    scores_df.loc[regressorName + '_' + dataType, 'f1'] = scores['test_f1'].mean()
    scores_df.loc[regressorName + '_' + dataType, 'roc_auc'] = scores['test_roc_auc'].mean()
    scores_df.loc[regressorName + '_' + dataType, 'recall'] = scores['test_recall'].mean()
    scores_df.loc[regressorName + '_' + dataType, 'average_precision'] = scores['test_average_precision'].mean()


def cross_val_all_regerssors(X_train,y_train,feature_set):
    global scores_df
    for regr in regressors_type:
        get_cross_val_score(regr, X_train, y_train,feature_set)

def fit_predict_all_regressors(X_train,y_train,X_test):
    global regressors_type
    global regr_result_db
    for reg_type  in regressors_type:
        regr = reg_type
        regr.fit(X_train,y_train)
        save_estimator(regr)
        regressors_prediction[regr.__class__.__name__]= regr.predict(X_test)

### Get all possible features of document after pasring ###
def save_doc_features(dir_name,doc_idx,ft):
    doc_name = os.path.join(os.getcwd(),defines.PATH_TO_DFS,dir_name,"{:02d}_sent_pos_db.csv".format(doc_idx))
    if not os.path.isfile(doc_name):
        print("ERROR: {} does not exists".format(doc_name))
        return
    get_and_save_sent_lemma_db(dir_name,doc_idx)
    get_and_save_sent_pos_count_db(dir_name,doc_idx)
    merge_sent_pos_db(dir_name,doc_idx)
    get_and_save_sent_vectors(dir_name,doc_idx,ft)
    get_and_save_doc_similarity(dir_name,doc_idx)


######

def leave_out_validate(X,y,groups,logo):
    score_list = []
    for train_idx, test_idx in logo.split(X, y, groups):
        score_list.append(get_prediction(train_idx,test_idx,X,y))
    return np.array(score_list)


def get_prediction(train_idx,test_idx,X,y):
    crf = CRF(
    min_freq = 5,
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True,
    )
#     print ("Test idx: {}..., train idx: {}...".format(train_idx[:10],test_idx[:10]))
    X_train = itemgetter(*train_idx)(X)
    y_train  = itemgetter(*train_idx)(y)
    X_test = itemgetter(*test_idx)(X)
    y_test = itemgetter(*test_idx)(y)
    
    # sample_features(y_test)
    print("True labels {:.2f} of train, {:.2f} of test".format(count_true_labels_ratio(y_train),count_true_labels_ratio(y_test)))
    
    crf.fit(X_train, y_train)
    y_pred  =  crf.predict(X_test)
    labels = list(crf.classes_)
    f1 = metrics.flat_f1_score(y_test, y_pred,average='weighted', labels=labels)
    recall = metrics.flat_recall_score(y_test, y_pred,average='weighted', labels=labels)
    precision = metrics.flat_precision_score(y_test, y_pred,average='weighted', labels=labels)
    return [f1,recall,precision]

def add_score_to_db(score_db,prefix,score):
    mean_values = score.mean(axis=0)
    print("mean_values {}".format(mean_values))
    score_db.loc[prefix,"f1"] = mean_values[0]
    score_db.loc[prefix,"recall"] = mean_values[1]
    score_db.loc[prefix,"precision"] = mean_values[2]
    return score_db

def count_true_labels_ratio(in_list): 
    # Input: list of lists
    # Output: ratio of True labels
    total_len = sum( [ len(listElem) for listElem in in_list])
    total_true = sum( [ listElem.count("True") for listElem in in_list])
    return total_true/total_len