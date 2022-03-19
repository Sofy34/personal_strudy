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
from sklearn.metrics import ConfusionMatrixDisplay
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
ft = fasttext.load_model('./external_src/cc.he.300.bin')


### EMBEDDED VECTORS ###

def get_and_save_sent_vectors(doc_idx,dim = 300): 
    sent_db = pd.read_csv(os.path.join(os.getcwd(),defines.PATH_TO_DFS,"{:02d}_sent_db.csv".format(doc_idx)))
    sent_vec_db = get_vector_per_sentence(sent_db,dim)
    sent_vec_db.to_csv(os.path.join(os.getcwd(),defines.PATH_TO_DFS,"{:02d}_sent_vec{}_db.csv".format(doc_idx,dim)),index=False)
    print("{} doc sent saved".format(doc_idx,dim))

def get_vector_per_sentence(db, dim = 300):
    global ft
    if (dim < 300):
        fasttext.util.reduce_model(ft, dim)
    sent_vectors = [ft.get_sentence_vector(row['text']) for index, row in db.iterrows()]
    # sent_array = np.vstack(sent_vectors)
    sent_vec_db = pd.DataFrame(sent_vectors)
    return sent_vec_db

def get_and_save_doc_similarity(doc_idx,dim = 300): 
    sent_vec_db = pd.read_csv(os.path.join(os.getcwd(),defines.PATH_TO_DFS,"{:02d}_sent_vec{}_db.csv".format(doc_idx,dim)))
    sim_db = pd.DataFrame(cosine_similarity(sent_vec_db))
    sim_db.to_csv(os.path.join(os.getcwd(),defines.PATH_TO_DFS,"{:02d}_sent_sim_vec{}_db.csv".format(doc_idx,dim)),index=False)
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

def tfidf_transform_doc(doc_idx,tfidf):
    sent_lemma_db = pd.read_csv(os.path.join(os.getcwd(),defines.PATH_TO_DFS,"{:02d}_sent_lemma_db.csv".format(doc_idx)))
    doc_lemmas = sent_lemma_db['sent_lemma'].tolist()
    return tfidf.transform(doc_lemmas)


def tfidf_fit():
    all_docs_lemma = get_all_docs_lemma()
    tfidf = TfidfVectorizer(lowercase=False)
    return tfidf.fit(all_docs_lemma['sent_lemma'].tolist())

def tfidf_build_all_save_per_doc():
    vocab = tfidf_fit()
    sent_lemma_db_list = glob.glob(os.path.join(os.getcwd(),defines.PATH_TO_DFS, "*_sent_lemma_db.csv"))
    for i,doc_name in enumerate(sent_lemma_db_list):
        doc_prefix = get_doc_idx_from_name(doc_name)
        X = tfidf_transform_doc(doc_prefix,vocab)
        sparse.save_npz(os.path.join(os.getcwd(),defines.PATH_TO_DFS,"{:02d}_tfidf.npz".format(doc_prefix)), X)
        print("TfIdf {} saved".format(doc_prefix))





#########################

## POS from YAP ###

def get_and_save_sent_lemma_db(doc_idx):
    doc_name = os.path.join(os.getcwd(),defines.PATH_TO_DFS,"{:02d}_sent_pos_db.csv".format(doc_idx))
    if not os.path.isfile(doc_name):
        print("ERROR: {} does not exists".format(doc_name))
        return
    sent_pos_db = pd.read_csv(doc_name,usecols=['sent_idx','LEMMA'])
    sent_lemma_db = pd.DataFrame()
    sent_lemma_db['sent_lemma'] = sent_pos_db.groupby('sent_idx')['LEMMA'].apply(lambda x: "%s" % ' '.join(x)).tolist()
    sent_lemma_db.to_csv(os.path.join(os.getcwd(),defines.PATH_TO_DFS,"{:02d}_sent_lemma_db.csv".format(doc_idx)),index=False)
    print("{} sent lemma db saved".format(doc_idx))



def get_and_save_sent_pos_count_db(doc_idx):
    columns_to_count = ['POSTAG','f_gen','f_num','f_suf_gen','f_suf_num','f_suf_per','f_per','f_tense']
    sent_pos_db = pd.read_csv(os.path.join(os.getcwd(),defines.PATH_TO_DFS,"{:02d}_sent_pos_db.csv".format(doc_idx)))
    sent_pos_dummies = pd.get_dummies(sent_pos_db,columns=columns_to_count)
    sent_pos_dummies.fillna(value=0,inplace=True)
    count_db = sent_pos_dummies.groupby('sent_idx').sum()
    count_db['TOKEN'] = sent_pos_dummies.groupby('sent_idx')['TOKEN'].max()
    count_db.drop(['FROM','TO','doc_idx'],inplace=True,axis=1)
    count_db.to_csv(os.path.join(os.getcwd(),defines.PATH_TO_DFS,"{:02d}_sent_pos_count_db.csv".format(doc_idx)),index=False)
    print("{} sent count db saved".format(doc_idx))

#########################

### Merge all sentense features into single DB ###
def merge_sent_pos_db(doc_idx):
    sent_db = pd.read_csv(os.path.join(os.getcwd(),defines.PATH_TO_DFS,"{:02d}_sent_db.csv".format(doc_idx)),usecols=['is_nar','is_client','sent_len'])
    count_db = pd.read_csv(os.path.join(os.getcwd(),defines.PATH_TO_DFS,"{:02d}_sent_pos_count_db.csv".format(doc_idx)))
    merged_db =  pd.merge(sent_db,count_db, left_index=True,right_index=True,validate="one_to_one")
    merged_db.to_csv(os.path.join(os.getcwd(),defines.PATH_TO_DFS,"{:02d}_merged_db.csv".format(doc_idx)),index=False)
    print("{} sent features db saved".format(doc_idx))

#########################

### Pack sentense features for CRF  ###

curr_doc_db = {}
def load_doc_features(doc_idx):
    global curr_doc_db
    curr_doc_db['merged'] = pd.read_csv(os.path.join(os.getcwd(),defines.PATH_TO_DFS,"{:02d}_merged_db.csv".format(doc_idx)))
    curr_doc_db['sim_vec']  = pd.read_csv(os.path.join(os.getcwd(),defines.PATH_TO_DFS,"{:02d}_sent_sim_vec300_db.csv".format(doc_idx)))
    curr_doc_db['tfidf'] = sparse.load_npz(os.path.join(os.getcwd(),defines.PATH_TO_DFS,"{:02d}_tfidf.npz".format(doc_idx)))
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
        doc_idx = get_doc_idx_from_name(doc_name)
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

def reshape_doc_features_to_sequence(X,y,groups,seq_len,step):
    X_seq = [X[i : i+seq_len] for i in np.arange(0,len(X),step)]
    y_seq = [y[i : i+seq_len] for i in np.arange(0,len(y),step)]
    groups_seq = groups[::step]
    print ("{} doc sentences reshaped: from {} to {}".format(groups_seq[0],len(X),len(X_seq)))
    return X_seq,y_seq,groups_seq

def pack_reshape_all_doc_sentences(seq_len,step):
    X = []
    y = []
    groups = []
    doc_db_path = os.path.join(os.getcwd(),defines.PATH_TO_DFS,"doc_db.csv")
    if os.path.isfile(doc_db_path):
        doc_db = pd.read_csv(doc_db_path)
    for doc_idx in doc_db.doc_idx_from_name:
        load_doc_features(doc_idx)
        X_doc,y_doc,groups_doc =  pack_doc_sentences(doc_idx)
        X_seq,y_seq,groups_seq = reshape_doc_features_to_sequence(X_doc,y_doc,groups_doc,seq_len,step)
        X.extend(X_seq)
        y.extend(y_seq)
        groups.extend(groups_seq)
        
    print("Sentenced packed for {} docs".format(len(doc_db.index)))
    return X,y,groups

def pack_all_doc_sentences():
    X = []
    y = []
    groups = []
    sent_lemma_db_list = glob.glob(os.path.join(os.getcwd(),defines.PATH_TO_DFS, "*_sent_lemma_db.csv"))
    # doc_db_path = os.path.join(os.getcwd(),defines.PATH_TO_DFS,"doc_db.csv")
    # if os.path.isfile(doc_db_path):
    #     doc_db = pd.read_csv(doc_db_path)
    for doc_name in sent_lemma_db_list:
        doc_idx = get_doc_idx_from_name(doc_name)
        load_doc_features(doc_idx)
        X_doc,y_doc,groups_doc =  pack_doc_sentences(doc_idx)
        X.extend(X_doc)
        y.extend(y_doc)
        groups.extend(groups_doc)
        
    print("{} sentenced packed for {} docs".format(len(X),len(sent_lemma_db_list)))
    return X,y,groups

def sent2features(sent_idx,idx_in_seq,seq_len=6,neighbor_radius =2,columns_start_idx = 1):
    global curr_doc_db
    features = {}
    for col in curr_doc_db['merged'].columns[columns_start_idx:]:
        features["{}".format(col)]= curr_doc_db['merged'].loc[sent_idx,col]

    if idx_in_seq > 1:
        update = {}
        for col in curr_doc_db['merged'].columns[columns_start_idx:]:
            update["-1:{}".format(col)]=curr_doc_db['merged'].loc[sent_idx-1,col]
        features.update(update)

    
    if idx_in_seq > 2:
        update = {}
        for col in curr_doc_db['merged'].columns[columns_start_idx:]:
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
    tfidf_feature_indices = curr_doc_db['tfidf'][sent_idx,:].nonzero()[1]
    for i in tfidf_feature_indices:
        update["tf_{}".format(i)] = curr_doc_db['tfidf'][sent_idx,i]
    features.update(update)

    
    if idx_in_seq < seq_len-1:
        update = {}
        for col in curr_doc_db['merged'].columns[columns_start_idx:]:
            update["+1:{}".format(col)]=curr_doc_db['merged'].loc[sent_idx+1,col]
        features.update(update)


    return features

def sent2label(sent_idx):
    return "{}".format(curr_doc_db['merged'].loc[sent_idx,'is_nar'].astype(bool))


def doc2features(first_sent_idx,seq_len=6,seq_step=6):
    return [sent2features(sent_idx,idx_in_seq) for idx_in_seq,sent_idx in enumerate(np.arange(first_sent_idx,first_sent_idx+seq_len,dtype=int))]
def doc2labels(first_sent_idx,seq_len=6,seq_step=6):
    return [sent2label(sent_idx) for sent_idx in range(first_sent_idx,first_sent_idx+seq_len)]

#########################

def get_doc_idx_from_name(file_name):
    base_name = os.path.basename(file_name)
    return int(base_name.split('_')[0])

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


def get_prediction_report(y_test,y_pred):
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred,cmap='gray_r')
    print(classification_report(y_test, y_pred))

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
def save_doc_features(doc_idx):
    doc_name = os.path.join(os.getcwd(),defines.PATH_TO_DFS,"{:02d}_merged_db.csv".format(doc_idx))
    if not os.path.isfile(doc_name):
        print("ERROR: {} does not exists".format(doc_name))
        return
    get_and_save_sent_lemma_db(doc_idx)
    get_and_save_sent_pos_count_db(doc_idx)
    merge_sent_pos_db(doc_idx)
    get_and_save_sent_vectors(doc_idx)
    get_and_save_doc_similarity(doc_idx)


######
