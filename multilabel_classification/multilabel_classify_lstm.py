# -*- coding: utf-8 -*-
"""
Created on Thu May  3 16:18:21 2018

@author: Administrator


Training Data Structure:
    'category','WYSText'
    脂肪肝 钙化灶  ,肝脏大小形态尚可....
    子宫肌瘤 宫颈囊肿  ,膀胱充盈壁光滑...
    脂肪肝 肾盂旁囊肿  ,肝脏回声前方...
    
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import random
from tensorflow.python.ops import rnn_cell
from sklearn import metrics

## Helper methods for reading and creating sequences of data for RNN/LSTM
def read_data(file_path):
    data=pd.read_csv(file_path, header=0, encoding='utf8')
    return data

def build_label_dict(data):
    label_dict={}
    for i in range(len(data)):
        for label in data.iloc[i,0].split(' '):
            if label not in label_dict.keys():
                label_dict[label]=len(label_dict.keys())
    return label_dict

def get_vocab(vocab_path):
    word2id={}
    df=pd.read_csv(vocab_path, sep=' ', header=None)
    for i in range(len(df)):
        word2id[df.iloc[i,0]]=i+1
    word2id['<UNK>']=len(df)+1
    word2id['<PAD>']=0
    return word2id

def sentence2id(sent, word2id):
    sentence_id = []
    for word in sent:
        if word.isdigit():
            word = '<NUM>'
        elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
            word = '<ENG>'
        if word not in word2id:
            word = '<UNK>'
        sentence_id.append(word2id[word])
    return sentence_id

def batch_yield(segments, labels, batch_size, vocab, label_dict, sequence_length, shuffle=False):
    if shuffle:
        random.shuffle(data)
    seqs, cates=[], []
    for i in segments.index:
        label_=[0]*len(label_dict.items())
        for label in labels[i].strip(' ').split(' '):
            label_[label_dict[label]]=1
        sent_=sentence2id(segments[i], vocab)
        if len(sent_)<=sequence_length:
            sent_=sent_+[0]*(sequence_length-len(sent_))
        else:
            sent_=sent_[:sequence_length]
        if len(seqs)==batch_size:
            yield np.array(seqs), np.array(cates)
            seqs, cates=[], []
        seqs.append(sent_)
        cates.append(label_)




## Import data
data=read_data('data_path/labeled_text_train2.csv')
label_dict=build_label_dict(data)
word2id=get_vocab('data_path/vocab.txt')
_embeddings = pd.read_csv('data_path/vectors.txt', encoding='utf8', sep=' ')
embeddings = np.array(_embeddings.iloc[:,1:], dtype='float32')


labels=data['category']
segments=data['WYSText']
train_test_spilt=np.random.rand(len(segments))<0.80
train_x=segments[train_test_spilt]
train_y=labels[train_test_spilt]
test_x=segments[~train_test_spilt]
test_y=labels[~train_test_spilt]



## Hyperparameters Configuration
tf.reset_default_graph()
learning_rate=0.001
training_epochs=100
batch_size=10
total_batches=(train_x.shape[0]//batch_size)

n_hidden=64
n_classes=len(label_dict.items())
alpha=0.5
sequence_length=100



## Input/Output placeholders for Tensorflow graph
x=tf.placeholder('float', [None, sequence_length, embeddings.shape[1]])
y=tf.placeholder('float', [None, n_classes])
y_steps=tf.placeholder('float', [None, n_classes])
sequence_lengths=tf.placeholder(tf.int32, shape=[None], name='sequence_lengths')
input_ids=tf.placeholder(dtype=tf.int32, shape=[None, None], name='input_ids')



## Helper methods for building LSTM network
def weight_variable(shape):
    initial=tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial=tf.constant(0.0, shape=shape)
    return tf.Variable(initial)

def LSTM(x, weight, bias):
    layers=[rnn_cell.LSTMCell(n_hidden, state_is_tuple=True) for _ in range(2)]
    multi_layer_cell=tf.nn.rnn_cell.MultiRNNCell(layers)
    # x shape=(?, 100(words), 100(embeddings))
    # outputs shape=(?, 100(words), 64(units))
    outputs, state=tf.nn.dynamic_rnn(cell=multi_layer_cell, inputs=x, sequence_length=sequence_lengths, dtype=tf.float32)
    # output_flattened shape=(?, 64) 将(?, 100, 64)展开成(?*100, 64)
    output_flattened=tf.reshape(outputs, [-1, n_hidden])
    # output_logits shape=(?*100, n_classes)
    output_logits=tf.add(tf.matmul(output_flattened, weight), bias)
    # output_all shape=(?*100, n_classes)
    output_all=tf.nn.sigmoid(output_logits)
    # output_reshaped shape=(?, 100, n_classes)
    output_reshaped=tf.reshape(output_all, [-1, sequence_length, n_classes])
    # output_last shape=(?, n_classes)
    output_last=tf.gather(tf.transpose(output_reshaped, [1, 0, 2]), sequence_length-1)
    return output_last, output_all

## Build word embeddings
input_word_embeddings=tf.Variable(embeddings, dtype=tf.float32, trainable=False, name='input_word_embeddings')
word_embeddings=tf.nn.embedding_lookup(params=input_word_embeddings, ids=input_ids, name='word_embeddings')

weight=weight_variable([n_hidden, n_classes])
bias=bias_variable([n_classes])
y_last, y_all=LSTM(x, weight, bias)

### Loss function: Binary cross entropy and target replication
all_steps_cost=-tf.reduce_mean((y_steps*tf.log(y_all))+ (1-y_steps)*tf.log(1-y_all))
last_step_cost=-tf.reduce_mean((y*tf.log(y_last))+ ((1-y)*tf.log(1-y_last)))
loss_function=(alpha*all_steps_cost)+((1-alpha)*last_step_cost)
optimizer=tf.train.AdamOptimizer(learning_rate).minimize(loss_function)



## Training and testing the model
with tf.Session() as session:
    tf.global_variables_initializer().run()
    session.run(input_word_embeddings)
    roc_auc_score=[]
    for epoch in range(training_epochs):
        batches=batch_yield(train_x, train_y, batch_size, word2id, label_dict, sequence_length, shuffle=False)
        for step, (batch_x, batch_y) in enumerate(batches):
            batch_x_emb=session.run(word_embeddings, feed_dict={input_ids:batch_x})
            _, c=session.run([optimizer, loss_function], feed_dict={x:batch_x_emb, y:batch_y, y_steps:np.tile(batch_y,((sequence_length),1)), sequence_lengths:[100]*batch_size})
        batches_test=batch_yield(test_x, test_y, batch_size, word2id, label_dict, sequence_length, shuffle=False)
        for step, (batch_test_x, batch_test_y) in enumerate(batches_test):
            batch_test_x_emb=session.run(word_embeddings, feed_dict={input_ids:batch_test_x})
            batch_pred_y=session.run(y_last, feed_dict={x:batch_test_x_emb, sequence_lengths:[100]*batch_size})
            for i in range(batch_size):
                roc_auc=metrics.roc_auc_score(batch_test_y[i], batch_pred_y[i])
                roc_auc_score.append(roc_auc)
        print('ROC AUC SCORE:', np.mean(roc_auc_score))
