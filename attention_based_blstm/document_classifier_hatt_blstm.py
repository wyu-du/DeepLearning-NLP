# -*- coding: utf-8 -*-
"""
Created on Tue May 29 15:40:54 2018

@author: Administrator
"""

import json
import pickle
from collections import Counter
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers.wrappers import Bidirectional
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Convolution1D, MaxPooling1D, Flatten, concatenate, GlobalMaxPooling1D
from keras.layers import GlobalMaxPooling1D, GlobalAveragePooling1D, SpatialDropout1D ,TimeDistributed
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.constraints import maxnorm
from keras.models import model_from_json
from keras.optimizers import Adam
from keras import regularizers


import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from attention_blstm_model.attention import AttentionWithContext
from attention_blstm_model.data_gen import hierarchicalCorpus as Corpus


# Modify this paths as well
DATA_DIR = 'data_path/'
TRAIN_FILE = 'labeled_text_train2.csv'
EMBEDDING_FILE = 'data_path/vectors.txt'
# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 80000
# Max number of words in each abstract.
MAX_SEQUENCE_LENGTH = 100 # MAYBE BIGGER
MAX_SENT_LENGTH=10
MAX_SEQ_LEN=10
# This is fixed.
EMBEDDING_DIM = 100
# The name of the model.
STAMP = 'doc_hatt_blstm'

def f1_score(y_true, y_pred):
    y_true=tf.cast(y_true, 'float32')
    y_pred=tf.cast(tf.round(y_pred), 'float32')
    y_correct=y_true*y_pred
    y_correct=tf.where(tf.is_nan(y_correct), tf.zeros_like(y_correct), y_correct)
    
    sum_true=tf.reduce_sum(y_true, axis=1)
    sum_pred=tf.reduce_sum(y_pred, axis=1)
    sum_correct=tf.reduce_sum(y_correct, axis=1)
    
    precision=sum_correct/sum_pred
    precision=tf.where(tf.is_nan(precision), tf.zeros_like(precision), precision)
    recall=sum_correct/sum_true
    f_score=1*precision*recall/(precision+recall)
    f_score=tf.where(tf.is_nan(f_score), tf.zeros_like(f_score), f_score)
    return tf.reduce_mean(f_score)

def precision_score(y_true, y_pred):
    y_true=tf.cast(y_true, 'float32')
    y_pred=tf.cast(tf.round(y_pred), 'float32')
    y_correct=y_true*y_pred
    y_correct=tf.where(tf.is_nan(y_correct), tf.zeros_like(y_correct), y_correct)
    
    sum_pred=tf.reduce_sum(y_pred, axis=1)
    sum_correct=tf.reduce_sum(y_correct, axis=1)
    
    precision=sum_correct/sum_pred
    precision=tf.where(tf.is_nan(precision), tf.zeros_like(precision), precision)
    return tf.reduce_mean(precision)

def recall_score(y_true, y_pred):
    y_true=tf.cast(y_true, 'float32')
    y_pred=tf.cast(tf.round(y_pred), 'float32')
    y_correct=y_true*y_pred
    y_correct=tf.where(tf.is_nan(y_correct), tf.zeros_like(y_correct), y_correct)
    
    sum_true=tf.reduce_sum(y_true, axis=1)
    sum_correct=tf.reduce_sum(y_correct, axis=1)
    
    recall=sum_correct/sum_true
    return tf.reduce_mean(recall)

def load_data(train_set):
    X_data=[]
    y_data=[]
    for c, (vector, target) in enumerate(train_set):
        X_data.append(vector)
        y_data.append(target)
        if c%10000==0:
            print(c)
    print(len(X_data), 'training examples')
    
    class_freqs=Counter([y for y_seq in y_data for y in y_seq]).most_common()
    class_list=[y[0] for y in class_freqs]
    nb_classes=len(class_list)
    print(nb_classes, 'classes')
    class_dict=dict(zip(class_list, np.arange(len(class_list))))
    
    with open('data_path_save/attention_blstm/class_dict.pkl', 'wb') as fp:
        pickle.dump(class_dict, fp)
    print('Exported class dictionary')
    
    y_data_int=[]
    for y_seq in y_data:
        y_data_int.append([class_dict[y] for y in y_seq])
    
    X_data_flat=[]
    for raw_text in X_data:
        flat_text=[]
        for sent in raw_text:
            flat_text.extend(sent)
        X_data_flat.append(flat_text)
        
    tokenizer=Tokenizer(num_words=MAX_NB_WORDS, oov_token=1)
    tokenizer.fit_on_texts(X_data_flat)
    X_data_int=np.zeros((len(X_data), MAX_SEQ_LEN, MAX_SENT_LENGTH))
    for idx, raw_text in enumerate(X_data):
        sents_batch=np.zeros((MAX_SEQ_LEN, MAX_SENT_LENGTH))
        tokens=tokenizer.texts_to_sequences(raw_text)
        sents=pad_sequences(tokens, maxlen=MAX_SENT_LENGTH, padding='post', truncating='post', dtype='float32')
        for j, sent in enumerate(sents):
            if j>=MAX_SEQ_LEN:
                break
            sents_batch[j,:]=sent
        X_data_int[idx, :, :]=sents_batch
    X_data=X_data_int
    print('Shape of data tensor:', X_data.shape)
    
    word_index=tokenizer.word_index
    print('Found %s unique tokens'%len(word_index))
    with open('data_path_save/attention_blstm/word_index.json', 'w') as fp:
        json.dump(word_index, fp)
    print('Exported word dictionary')
    
    mlb=MultiLabelBinarizer()
    mlb.fit([class_dict.values()])
    y_data=mlb.transform(y_data_int)
    print('Shape of label tensor:', y_data.shape)
    
    X_train, X_val, y_train, y_val=train_test_split(X_data, y_data, train_size=0.8, test_size=0.2, random_state=42)
    
    return X_train, X_val, y_train, y_val, nb_classes, word_index

def prepare_embeddings(wrd2id):
    vocab_size=MAX_NB_WORDS
    print('Found %s words in the vocabulary.'%vocab_size)
    
    embedding_idx={}
    glove_f=open(EMBEDDING_FILE, encoding='utf8')
    for line in glove_f:
        values=line.split()
        wrd=values[0]
        coefs=np.asarray(values[1:], dtype='float32')
        embedding_idx[wrd]=coefs
    glove_f.close()
    print('Found %s word vectors.'%len(embedding_idx))
    
    embedding_mat=np.random.rand(vocab_size+1, EMBEDDING_DIM)
    wrds_with_embeddings=0
    for wrd, i in wrd2id.items():
        if i>vocab_size:
            continue
        embedding_vec=embedding_idx.get(wrd)
        if embedding_vec is not None:
            wrds_with_embeddings+=1
            embedding_mat[i]=embedding_vec
    print(embedding_mat.shape)
    print('Words with embeddings: ', wrds_with_embeddings)
    
    return embedding_mat, vocab_size

def build_model(nb_classes, word_index, embedding_dim, seq_length, stamp):
    embedding_matrix, nb_words=prepare_embeddings(word_index)
    input_layer=Input(shape=(MAX_SEQ_LEN, MAX_SENT_LENGTH), dtype='int32')
    sent_input=Input(shape=(MAX_SENT_LENGTH, ), dtype='int32')
    embedding_layer=Embedding(input_dim=nb_words+1, output_dim=embedding_dim, input_length=MAX_SENT_LENGTH, 
                              weights=[embedding_matrix], embeddings_regularizer=regularizers.l2(0.00), 
                              trainable=True)(sent_input)
    drop1=SpatialDropout1D(0.3)(embedding_layer)
    sent_lstm=Bidirectional(LSTM(100, name='blstm_1', activation='tanh', recurrent_activation='hard_sigmoid',
                                 recurrent_dropout=0.0, dropout=0.4, kernel_initializer='glorot_uniform',
                                 return_sequences=True), merge_mode='concat')(drop1)
    sent_att_layer=AttentionWithContext()(sent_lstm)
    sentEncoder=Model(sent_input, sent_att_layer)
    sentEncoder.summary()
    
    textEncoder=TimeDistributed(sentEncoder)(input_layer)
    drop2=Dropout(0.4)(textEncoder)
    lstm_1=Bidirectional(LSTM(100, name='blstm_2', activation='tanh', recurrent_activation='hard_sigmoid',
                                 recurrent_dropout=0.0, dropout=0.4, kernel_initializer='glorot_uniform',
                                 return_sequences=True), merge_mode='concat')(drop2)
    lstm_1=BatchNormalization()(lstm_1)
    att_layer=AttentionWithContext()(lstm_1)
    drop3=Dropout(0.5)(att_layer)
    predictions=Dense(nb_classes, activation='sigmoid')(drop3)
    model=Model(inputs=input_layer, outputs=predictions)
    adam=Adam(lr=0.001, decay=0.0)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=[precision_score, recall_score, f1_score])
    model.summary()
    print(stamp)
    
    # save the model
    model_json=model.to_json()
    with open(stamp + ".json", "w") as json_file:
        json_file.write(model_json)
    return model

def load_model(stamp):
    json_file=open('data_path_save/attention_blstm/'+stamp+'.json', 'r')
    loaded_model_json=json_file.read()
    json_file.close()
    model=model_from_json(loaded_model_json, {'AttentionWithContext':AttentionWithContext})
    model.load_weights('data_path_save/attention_blstm/'+stamp+'.h5')
    print('Load model from disk')
    model.summary()
    adam=Adam(lr=0.001)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=[precision_score, recall_score, f1_score])
    return model

if __name__=='__main__':
    load_previous=False
    train_set=Corpus(DATA_DIR+TRAIN_FILE)
    X_train, X_val, y_train, y_val, nb_classes, word_index=load_data(train_set)
    
    if load_previous:
        model=load_model(STAMP)
    else:
        model=build_model(nb_classes, word_index, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH, STAMP)
    
    monitor_metric='val_f1_score'
    early_stopping=EarlyStopping(monitor=monitor_metric, patience=5)
    best_model_path='data_path_save/attention_blstm/'+STAMP+'.h5'
    model_checkpoint=ModelCheckpoint(best_model_path, monitor=monitor_metric, verbose=1,
                                     save_best_only=True, mode='max', save_weights_only=True)
    hist=model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32, shuffle=True, callbacks=[model_checkpoint])
    print(hist.history)