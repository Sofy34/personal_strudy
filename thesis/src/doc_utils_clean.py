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
from nltk import tokenize


pd.options.display.float_format = '{:f}'.format

global doc_db
global par_db
global block_db
global sent_db


def get_random_paragraph(query):
    match =  par_db.query(query)
    return match.sample()


def check_text_for_illegal_labels(par):
        # capture if there is [start:start] or [end:end]
    error_pattern_st = "(" + defines.START_CHAR + "(?:(?!" + defines.END_CHAR + ").)*" +  defines.START_CHAR +")";
    error_pattern_end = "(" + defines.END_CHAR + "(?:(?!" + defines.START_CHAR + ").)*" +  defines.END_CHAR +")";
    error_pattern = "(" + error_pattern_st + "|" +  error_pattern_end + ")";
    if (re.search(error_pattern,par)):
        print("\nstring gave ERROR\n{}".format(par))
        return 1
    return 0

def get_doc_idx_from_name(file_name):
    return int(file_name.split('_')[0])

def add_sent_column_for_labels():
    global sent_db
    sent_db['sent_idx_in_nar'] =  sent_db[sent_db['is_nar']==1].groupby(['doc_idx','nar_idx']).cumcount()+1
    sent_db['nar_len_in_sent'] =  sent_db[sent_db['is_nar']==1].groupby(['doc_idx','nar_idx'])['sent_idx_in_nar'].transform('max')

    sent_db['sent_idx_out_nar'] =  sent_db[sent_db['is_nar']==0].groupby(['doc_idx','block_idx']).cumcount()+1
    sent_db['fist_sent_in_nar'] =  np.where(sent_db['sent_idx_in_nar'] == 1, True, False)
    sent_db['last_sent_in_nar'] =  np.where(sent_db['sent_idx_in_nar'] == sent_db['nar_len_in_sent'], True, False)
    sent_db['is_client'] =  np.where(sent_db['par_type'] == 'client', 1, 0)

def split_block_to_sentences(text):
    text = remove_lr_annotation(text)
    text = remove_brackets(text) # important to remove before we split into sentences
    text = remove_symbols(text)

    sent_list = tokenize.sent_tokenize(text)
    
    for i,item in enumerate(sent_list):
        clean_item = clean_text(item)
        check_text_for_symbols(clean_item)
        if len(clean_item) != 0 and clean_item.isspace() == False: # disregard empty strings
            sent_list[i] = clean_item
    return sent_list


def check_text_for_symbols(text):
    if re.search(r'[\t,\\t]',text):
        print("ERROR text has bad symbols {}".format(text))
        quit()

def add_sentences_of_blocks_to_db(block_db_idx):
    global sent_db
    block_line = block_db.iloc[block_db_idx]
    block = block_line['text']
    sent_list = split_block_to_sentences(block)
    for i,sentence in enumerate(sent_list):
        curr_db_idx = sent_db.shape[0]
        sent_db.loc[curr_db_idx,'text'] = sentence
        sent_db.loc[curr_db_idx,'sent_idx_in_block'] = i
        sent_db.loc[curr_db_idx,'block_idx'] = block_db_idx
        sent_db.loc[curr_db_idx,'is_nar'] = block_line['is_nar']
        sent_db.loc[curr_db_idx,'doc_idx'] = block_line['doc_idx']
        sent_db.loc[curr_db_idx,'par_db_idx'] = block_line['par_db_idx']
        sent_db.loc[curr_db_idx,'par_idx_in_doc'] = block_line['par_idx_in_doc']
        sent_db.loc[curr_db_idx,'par_type'] = block_line['par_type']
        sent_db.loc[curr_db_idx,'block_type'] = block_line['block_type']
        sent_db.loc[curr_db_idx,'nar_idx'] = block_line['nar_idx']
        sent_db.loc[curr_db_idx,'sent_len'] = len(sentence)

        

def split_par_to_blocks_keep_order(par_db_idx):
    par = par_db.loc[par_db_idx,'text']
    startNum = par.count(defines.START_CHAR)
    endNum = par.count(defines.END_CHAR)
    block_list = [] # holds tupple ("tag", "block string")
    tag = ""
    outside_nar = ""
    splited = []

    if startNum == 0 and endNum == 0: # text is missing start and end symbols
        if par_db.loc[par_db_idx,'is_nar'] == 0: #entire paragraph is not narrative
            tag = "not_nar"
        else: #entire paragraph is narrative
            tag = "middle"
        block_list.insert(0,(tag,par))
    else:
        splited = re.split('(&|#)', par) # used for keeping original order between blocks
        splited_clean = splited
        for i,block in enumerate(splited):
            if '%' in block: # TBD handle story summary
                continue
            splited_clean[i] = clean_text(block)
        my_regex = {
            'whole' :defines.START_CHAR + ".*?" + defines.END_CHAR,     # [start:end] 
            'start' : defines.START_CHAR + ".*", # [start:]
            'end' :".*" +  defines.END_CHAR # [:end]
        }
        outside_nar = par
        for tag,regex in my_regex.items():
            nar_blocks = re.findall(regex,outside_nar)
            for j,block in enumerate(nar_blocks):
                if len(block) !=0:
                    block_idx = get_index_of_block_in_par(splited_clean,block,par_db_idx)
                    splited[block_idx] = "" # erase narrative blocks from splited paragraph
                    block_list.insert(block_idx,(tag,block))
            outside_nar = re.sub(regex,'',outside_nar)
        
        # handle the rest items in list - that must be non-narrative
        for i,block in enumerate(splited):
            if len(block)!=0:
                if '%' in block:
                    continue # TBD handle story summary
                block_idx = get_index_of_block_in_par(splited_clean,block,par_db_idx)
                block_list.insert(block_idx,("not_nar",block))
    return block_list

def get_index_of_block_in_par(splited,block,par_db_idx):
    cl_block = clean_text(block)
    if not cl_block in splited:
        print("{} \n par[{}]not in \n{}".format(par_db_idx,cl_block,splited))
        return -1
    else:
        return splited.index(cl_block)
    

def get_last_nar_idx_from_block_db():
    global block_db
    return block_db['nar_idx'].max()

def add_blocks_of_par_to_db(par_db_idx):
    global par_db, block_db
    is_nar = 0
    block_list = split_par_to_blocks_keep_order(par_db_idx)
    par_db_line = par_db.iloc[par_db_idx]
    for i,tupple in enumerate(block_list):
        curr_db_idx = block_db.shape[0]
        curr_nar_idx = 0 if curr_db_idx == 0 else get_last_nar_idx_from_block_db() 
        if tupple[0] in ['start','whole']:
            curr_nar_idx+=1
        is_nar = 1 if tupple[0] != 'not_nar' else 0
        block_db.loc[curr_db_idx,'text'] = tupple[1]
        block_db.loc[curr_db_idx,'is_nar'] = is_nar
        block_db.loc[curr_db_idx,'doc_idx'] = par_db_line['doc_idx']
        block_db.loc[curr_db_idx,'par_idx_in_doc'] = par_db_line['par_idx_in_doc']
        block_db.loc[curr_db_idx,'par_db_idx'] = par_db_idx
        block_db.loc[curr_db_idx,'par_type'] = par_db_line['par_type']
        block_db.loc[curr_db_idx,'block_type'] = tupple[0]
        block_db.loc[curr_db_idx,'nar_idx'] = curr_nar_idx if is_nar else 0

def read_csv(base_filename):
    return pd.read_csv("/".join([".",defines.PATH_TO_DFS, base_filename]))

def save_df_to_csv(df_name, is_global = True):
    path_to_save = "/".join([".",defines.PATH_TO_DFS,df_name])
    _df = globals().get(df_name,None)
    if _df is None:
        print("Can't find global df {}".format(df_name))
        return 1
    else:
        _df.to_csv("{}.csv".format(path_to_save),index=False)
        return 0



def save_doc_blocks(doc_idx):
    global par_db,block_db
    block_db = pd.DataFrame()
    doc_idx_from_name = doc_db.loc[doc_idx,'doc_idx_from_name']
    par_db = pd.read_csv(os.path.join(os.getcwd(),defines.PATH_TO_DFS,"{:02d}_par_db.csv".format(doc_idx_from_name)))
    for i in par_db.index:
        add_blocks_of_par_to_db(i)
    del par_db
    block_db.to_csv(os.path.join(os.getcwd(),defines.PATH_TO_DFS,"{:02d}_block_db.csv".format(doc_idx_from_name)),index=False)
    del block_db
    print("Doc {} blocks saved".format(doc_idx))

def save_doc_sentences(doc_idx):
    global block_db, sent_db
    sent_db = pd.DataFrame()
    doc_idx_from_name = doc_db.loc[doc_idx,'doc_idx_from_name']
    block_db = pd.read_csv(os.path.join(os.getcwd(),defines.PATH_TO_DFS,"{:02d}_block_db.csv".format(doc_idx_from_name)))
    for i in block_db.index:
        add_sentences_of_blocks_to_db(i)
    del block_db
    add_sent_column_for_labels()
    sent_db.to_csv(os.path.join(os.getcwd(),defines.PATH_TO_DFS,"{:02d}_sent_db.csv".format(doc_idx_from_name)),index=False)
    del sent_db
    print("Doc {} sentences saved".format(doc_idx))

def save_doc_paragraphs(doc_idx):
    global par_db, doc_db
    inside_narrative = 0
    doc = docx.Document(doc_db.loc[doc_idx,'path'])
    par_db = pd.DataFrame()
    doc_idx_from_name = doc_db.loc[doc_idx,'doc_idx_from_name']
    for i,par in enumerate(doc.paragraphs):
        curr_par_db_idx = par_db.shape[0]
        text,par_type = get_par_type_erase(par.text)
        if len(text) == 0:
            continue
        par_db.loc[curr_par_db_idx,'doc_idx'] = doc_idx_from_name
        par_db.loc[curr_par_db_idx,'text'] = text
        par_db.loc[curr_par_db_idx,'par_len'] = len(text)
        par_db.loc[curr_par_db_idx,'par_type'] = par_type
        par_db.loc[curr_par_db_idx,'par_idx_in_doc'] = i
        if defines.START_CHAR in par.text:
            inside_narrative = 1
        par_db.loc[curr_par_db_idx,'is_nar'] = inside_narrative
        if par.text.rfind(defines.END_CHAR) > par.text.rfind(defines.START_CHAR): # if [...# ] or [ ...&...#]
            inside_narrative = 0
    par_db.to_csv(os.path.join(os.getcwd(),defines.PATH_TO_DFS,"{:02d}_par_db.csv".format(doc_idx_from_name)),index=False)
    del par_db
    print("Doc {} paragraphs saved".format(doc_idx))

def save_all_docs_paragraphs():
    global doc_db
    for doc_idx in doc_db.index:
        save_doc_paragraphs(doc_idx)




def save_docs_db():
    global doc_db
    doc_db = pd.DataFrame()
    doc_path_list = get_labeled_files()
    for path in doc_path_list:
        add_doc_to_db(path)
    doc_db['doc_idx_from_name'] = doc_db['doc_idx_from_name'].astype(int)
    doc_db.to_csv(os.path.join(os.getcwd(),defines.PATH_TO_DFS,"doc_db.csv"),index=False)


def remove_punctuation(_text):
    text = _text.translate(str.maketrans('', '',string.punctuation))
    return text
 

def get_labeled_files():
    doc_path_list = []
    for file in glob.glob("./tmp/*_lc.docx"): # _lc is name pattern of labeled cleaned *.docx files
        doc_path_list.append(file)
    return doc_path_list


def add_doc_to_db(path):
    global doc_db
    file_name = os.path.basename(path)
    doc_db_idx = doc_db.shape[0]
    doc_idx_from_name = get_doc_idx_from_name(file_name)
    doc_db.loc[doc_db_idx,'path'] = path
    doc_db.loc[doc_db_idx,'file_name'] = file_name
    doc_db.loc[doc_db_idx,'doc_idx_from_name'] = doc_idx_from_name

def remove_brackets(text):
    return re.sub(r'\(\..*?\)|\[.*?\]','',text)

def clean_text(text):
    text,_ =  extract_narrative_summary(text)
    text = remove_punctuation(text)
    return text

def remove_symbols(text):
    return re.sub(r'[@#&\*\t\\t]','',text)

def extract_narrative_summary(text):
    summary = re.findall('%.*?%',text)
    for i in summary:
        text = re.sub(i,'',text)
    return text,summary

def remove_lr_annotation(text):
    return re.sub(r'\(L[0-9].*?\-[A-Z]{1}\)','',text)


def get_par_type_erase(par):
    par_type = 'no_mark'
    if 'CLIENT' in par[:20]: # search for a tag in the begginning of a line
        par = re.sub('CLIENT(:)', '',par)
        par_type = 'client'
    if 'THERAPIST' in par[:20]: # search for a tag in the begginning of a line
        par = re.sub('THERAPIST(:)', '',par)
        par_type= 'therapist'
    check_text_for_illegal_labels(par)
    return par,par_type

def get_all_data():
    save_docs_db()
    for doc_idx in doc_db.index:
        save_doc_paragraphs(doc_idx)
        save_doc_blocks(doc_idx)
        save_doc_sentences(doc_idx)
    
    


