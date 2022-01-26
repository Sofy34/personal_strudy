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

regressors = [
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
    X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=101,stratify=y.name)
    return X_train, X_test, y_train, y_test

def get_random_par(db,is_nar,len_threshold=30):
    return db.query("is_nar==1 & par_len >= @len_threshold").sample(n=1)


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
        global scores_df
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
        
def add_score(scores, regressorName, dataType):
    global scores_df
    scores_df.loc[regressorName + '_' + dataType, 'f1'] = scores['test_f1'].mean()
    scores_df.loc[regressorName + '_' + dataType, 'roc_auc'] = scores['test_roc_auc'].mean()
    scores_df.loc[regressorName + '_' + dataType, 'recall'] = scores['test_recall'].mean()
    scores_df.loc[regressorName + '_' + dataType, 'average_precision'] = scores['test_average_precision'].mean()


def run_on_all_regerssors(X_train,y_train,feature_set):
    global scores_df
    for regr in regressors:
        get_cross_val_score(regr, X_train, y_train,feature_set)