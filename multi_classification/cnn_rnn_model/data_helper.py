# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 10:14:06 2018

@author: Administrator
"""

import re
import logging
import numpy as np
import pandas as pd
from collections import Counter
import jieba
jieba.load_userdict('data_path/user_dict.txt')

logging.getLogger().setLevel(logging.INFO)

def clean_str(s):
    r='[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`ò¢{|}~！，。“”、\n：\tⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ（）【】～； \r\n]'
    s=re.sub(r,'',s)
    seg_list=jieba.cut(s,cut_all=False)
    out_list=[]
    for seg in seg_list:
        out_list.append(seg)
    out_s=' '.join(out_list)
    return out_s


def load_embeddings(vocabulary):
    word_embeddings={}
    for word in vocabulary:
        word_embeddings[word]=np.random.uniform(-0.25, 0.25, 300)
    return word_embeddings


def pad_sentences(sentences, padding_word='<PAD/>', forced_sequence_length=None):
    # Training 
    if forced_sequence_length is None:
        sequence_length=max(len(x) for x in sentences)
    # Predicting
    else:
        logging.critical('This is prediction, reading the trained sequence length')
        sequence_length=forced_sequence_length
    logging.critical('The maximum length is {}'.format(sequence_length))
    
    padded_sentences=[]
    for i in range(len(sentences)):
        sentence=sentences[i]
        num_padding=sequence_length-len(sentence)
        
        # Prediction: cut off the sentence if it is longer than the sequence length
        if num_padding<0:
            logging.info('This sentence has to be cut off because it is longer than trained sequence length')
            padded_sentence=sentence[0: sequence_length]
        else:
            padded_sentence=sentence+[padding_word]*num_padding
        padded_sentences.append(padded_sentence)
    return padded_sentences


def build_vocab(sentences):
    word_counts={}
    for sent in sentences:
        for word in sent:
            if word not in word_counts.keys():
                word_counts[word]=1
            else:
                word_counts[word]+=1
    word_counts=Counter(word_counts)
    vocabulary_inv=[word[0] for word in word_counts.most_common()]
    vocabulary={word:index for index,word in enumerate(vocabulary_inv)}
    return vocabulary, vocabulary_inv


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data=np.array(data)
    data_size=len(data)
    num_batches_per_epoch=int(data_size/batch_size)+1
    
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices=np.random.permutation(np.arange(data_size))
            shuffled_data=data[shuffle_indices]
        else:
            shuffled_data=data
        for batch_num in range(num_batches_per_epoch):
            start_index=batch_num*batch_size
            end_index=min((batch_num+1)*batch_size, data_size)
            yield shuffled_data[start_index:end_index]
            
           
def load_data(filename):
    df=pd.read_csv(filename, encoding='utf8')
    selected=['category','WYSText']
    non_selected=list(set(df.columns)-set(selected))
    
    df=df.drop(non_selected, axis=1)
    df=df.dropna(axis=0, how='any', subset=selected)
    df=df.reindex(np.random.permutation(df.index))
    
    labels=sorted(list(set(df[selected[0]].tolist())))
    labels.append('<UNK>')
    num_labels=len(labels)
    one_hot=np.zeros((num_labels, num_labels), int)
    np.fill_diagonal(one_hot, 1)
    label_dict=dict(zip(labels, one_hot))
    
    x_raw=df[selected[1]].apply(lambda x:clean_str(x).split(' ')).tolist()
    y_raw=df[selected[0]].apply(lambda y:label_dict[y]).tolist()
    
    x_raw=pad_sentences(x_raw)
    vocabulary, vocabulary_inv=build_vocab(x_raw)
    
    x=np.array([[vocabulary[word] for word in sentence] for sentence in x_raw])
    y=np.array(y_raw)
    return x, y, vocabulary, vocabulary_inv, df, labels

if __name__=='__main__': 
    train_file='labeled_text.csv'
    x, y, vocabulary, vocabulary_inv, df, labels=load_data(train_file)
