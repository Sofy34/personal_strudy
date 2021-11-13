#!pip install python-docx
import docx
import os, sys
import glob
import re
import string
import defines

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, PassiveAggressiveClassifier, Perceptron, RidgeClassifier, RidgeClassifierCV, SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer , TfidfVectorizer, TfidfTransformer
from sklearn.preprocessing import normalize, StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline, FeatureUnion

from bidi import algorithm as bidialg      # needed for arabic, hebrew
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
pd.options.display.float_format = '{:f}'.format

# utils for files 

def remove_punctuation(text):
    return text.translate(str.maketrans('', '',string.punctuation))
 

def get_labeled_files():
    doc_path_list = []
    for file in glob.glob("tmp/*_l.docx"): # _l is name pattern of labeled *.docx files
        doc_path_list.append(file)
    return doc_path_list

def get_doc_objects(doc_path_list):
    doc_list = []
    for path in doc_path_list: 
        print(path)
        doc_list.append(docx.Document(path))
    return doc_list

def add_doc_to_db(path,doc_db):
    print(path)
    file_name = os.path.basename(path)
    doc = docx.Document(path)
    client_tag, therapist_tag = get_client_therapist_tag(doc)
    num_par = len(doc.paragraphs)
    doc_list = [path,file_name,client_tag,therapist_tag,num_par]
    doc_db.loc[doc_db.shape[0]] = doc_list

# utils for doc content

# def get_client_therapist_tag(doc):
#     client_tag = ''
#     therapist_tag = ''
#     for par in doc.paragraphs[:20]:
#         if 'משתתפים' in par.text:
#             text = remove_punctuation(par.text)
#             split_par = text.split()
#             print("DEBUG ",split_par)
#             client_idx = split_par.index(defines.CLIENT_HEB)
#             therapist_idx = split_par.index(defines.THERAPIST_HEB)
#             client_tag = split_par[client_idx-1]
#             therapist_tag = split_par[therapist_idx-1]
#             break
#     return client_tag,therapist_tag

def get_client_therapist_tag(doc):
    client_tag = ''
    therapist_tag = ''
    for par in doc.paragraphs[:4]:
        if 'CLIENT' in par.text:
            client_tag = par.text.split()[0]
        if 'THERAPIST' in par.text:
            therapist_tag = par.text.split()[0]
    return client_tag,therapist_tag

def add_length_of_paragraphs(_df):
    df = _df.copy()
    df['par_len'] = df['text'].str.len()
    return df    

def clean_text(_df):
    df= _df.copy()
#     df['text'] = df['text'].str.replace(r'\n', '')
    df['text'] = df['text'].str.replace(r'[@,#,\*,\t]*','')
#     df['text'] = df['text'].str.replace(r'[a-zA-Z\u0590-\u05FF\u200f\u200e ]+$','')
    df['text']=remove_punctuation(df['text'].str)
    return df

def get_par_type_erase(par,doc_idx,doc_db):
    client_tag = doc_db.loc[doc_idx,'client_tag']
    therapist_tag = doc_db.loc[doc_idx,'therapist_tag']
#     segment_string = "".join([defines.SEGMENT_HEB,".*[0-9]+"])
    segment_string = "".join(["סגמנט",".*[0-9]"])
    if client_tag in par:
        par = par.replace(client_tag, '')
        return par,'client'
    if therapist_tag in par:
        par = par.replace(therapist_tag, '')
        return par,'therapist'
    if re.search(segment_string,par):
        return par,'segment'
    return par,'no_mark'

def add_paragraphs_to_db(doc_idx,doc_db,par_db):
    doc = docx.Document(doc_db.loc[doc_idx,'path'])
    inside_narrative  = 0
    narrative_idx = -1
    idx_in_nar = -1
    pre_par_speaker = ''
    for i,par in enumerate(doc.paragraphs):
        curr_par_db_idx = par_db.shape[0]
        text,par_type = get_par_type_erase(par.text,doc_idx,doc_db)
        par_db.loc[curr_par_db_idx,'doc_idx'] = doc_idx
        par_db.loc[curr_par_db_idx,'text'] = text
        par_db.loc[curr_par_db_idx,'par_type'] = par_type
        if par_type == 'no_mark':
            if curr_par_db_idx-1 in par_db.index and par_db.loc[curr_par_db_idx-1,'par_type']=='segment':
                par_db.loc[curr_par_db_idx,'par_type'] = par_db.loc[curr_par_db_idx-2,'par_type']       
        if par_db.loc[curr_par_db_idx,'par_type'] in ['segment','no_mark']:
            par_db.loc[curr_par_db_idx,'is_nar'] = 0
            print("skipping for doc {} entry {} type {} text {}".format(doc_idx,curr_par_db_idx,par_db.loc[curr_par_db_idx,'par_type'],text))
            continue # skip parahraph processing
        if defines.START_CHAR in par.text:
            inside_narrative = 1
            idx_in_nar = 0
            narrative_idx+=1
        par_db.loc[curr_par_db_idx,'is_nar'] = inside_narrative
        par_db.loc[curr_par_db_idx,'nar_idx'] = narrative_idx if (inside_narrative) else None
        par_db.loc[curr_par_db_idx,'idx_in_nar'] = idx_in_nar if (inside_narrative) else None
        idx_in_nar+=1
        if defines.END_CHAR in par.text:
            print ("update nar {} len to {}".format(narrative_idx,idx_in_nar))
            par_db.loc[par_db['nar_idx']==narrative_idx,'nar_len'] = idx_in_nar
            inside_narrative = 0


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

    
def sample_features(features):
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
    
def split_and_get_text(X,y):
    print ("total data len: {}".format(len(y)))
    X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=101,stratify=y)
    return X_train, X_test, y_train, y_test

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

def get_label_and_drop(_df):
    df = _df.copy()
    label = df['label']
    df = drop_columns(df,['label'])
    return df, label

def show_random_text(_df,feature,n=1):
    df = _df.sample(n=n,random_state=42)
    print(list(df[feature]))
    
def get_cross_val_score(scores_df,estimator,X_train,y_train,prefix="",sampler=None):
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
        add_score(scores_df, full_scores, estimator.__class__.__name__,prefix)
        
def add_score(scores_df, scores, regressorName, dataType):
    scores_df.loc[regressorName + '_' + dataType, 'f1'] = scores['test_f1'].mean()
    scores_df.loc[regressorName + '_' + dataType, 'roc_auc'] = scores['test_roc_auc'].mean()
    scores_df.loc[regressorName + '_' + dataType, 'recall'] = scores['test_recall'].mean()
    scores_df.loc[regressorName + '_' + dataType, 'average_precision'] = scores['test_average_precision'].mean()