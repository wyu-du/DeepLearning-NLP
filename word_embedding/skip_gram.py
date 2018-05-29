# -*- coding: utf-8 -*-
"""
Created on Thu May 24 14:19:59 2018

@author: Administrator
"""

import collections
import math
import numpy as np
import os
import random
import tensorflow as tf
import zipfile
from matplotlib import pylab
from six.moves import range
from six.moves.urllib.request import urlretrieve
from sklearn.manifold import TSNE
import pickle


# Download the data from the source website if necessary
url="http://mattmahoney.net/dc/"
def maybe_download(filename,expected_bytes):
    if not os.path.exists(filename):
        filename, _=urlretrieve(url+filename,filename)
    statinfo=os.stat(filename)
    if statinfo.st_size==expected_bytes:
        print("① Found and verfied ",filename)
    else:
        print(statinfo.st_size)
        raise Exception("Failed to verify ",filename,".Download with a broser!")
    return filename
filename=maybe_download("text8.zip",31344016)


# Read the data into a string
def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        data=tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data
words=read_data(filename)
print("② Data size is ",len(words))


# Build the dictionary and replace rare words with UNK token
vocabulary_size=50000
def build_dataset(words):
    count=[["UNK",-1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size-1))
    dictionary=dict()
    for word,_ in count:
        dictionary[word]=len(dictionary)
    data=list()
    unk_count=0
    for word in words:
        if word in dictionary:
            idx=dictionary[word]
        else:
            idx=0   # dictionary['UNK'], words that are not frequently occur
            unk_count+=1
        data.append(idx)
    count[0][1]=unk_count
    reverse_dictionary=dict(zip(dictionary.values(),dictionary.keys()))
    return data,count,dictionary,reverse_dictionary
data, count, dictionary, reverse_dictionary = build_dataset(words)
print('Most common words (+UNK)', count[:5])
del words  # Hint to reduce memory.


# Function to generate a training batch for the skip-gram model
data_index=0
def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size%num_skips==0
    assert num_skips<=2*skip_window
    batch=np.ndarray(shape=(batch_size), dtype=np.int32)
    labels=np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span=2*skip_window+1
    # deque是双端队列，可在队列两端添加和删除元素
    buffer=collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index=(data_index+1)%len(data)
    for i in range(batch_size//num_skips):
        target=skip_window
        targets_to_avoid=[skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target=random.randint(0, span-1)
            targets_to_avoid.append(target)
            batch[i*num_skips+j]=buffer[skip_window]
            labels[i*num_skips+j, 0]=buffer[target]
        buffer.append(data[data_index])
        data_index=(data_index+1)%len(data)
    return batch, labels


print('data:', [reverse_dictionary[di] for di in data[:8]])
for num_skips, skip_window in [(2, 4), (4, 2)]:
    data_index=0
    batch, labels=generate_batch(batch_size=8, num_skips=num_skips, skip_window=skip_window)
    print('\nwith num_skips = %d and skip_window = %d:' % (num_skips, skip_window))
    print('    batch:', [reverse_dictionary[bi] for bi in batch])
    print('    labels:', [reverse_dictionary[li] for li in labels.reshape(8)])
    


# Train a skip-gram model
batch_size=128
embeddings_size=128
skip_window=1       # How many words to consider left and right
num_skips=2         # How many times to reuse an input to generate a label
valid_size=16       # Random set of words to evaluate similarity on
valid_window=100    # Only pick dev samples in the head of the distribution
valid_examples=np.array(random.sample(range(valid_window), valid_size))
num_sampled=64      # Number of negative examples to sample

# Input data
train_dataset=tf.placeholder(tf.int32, shape=[batch_size])
train_labels=tf.placeholder(tf.int32, shape=[batch_size, 1])
valid_dataset=tf.constant(valid_examples, dtype=tf.int32)

# Variables
embeddings=tf.Variable(tf.random_uniform([vocabulary_size, embeddings_size], -1.0, 1.0))
softmax_weights=tf.Variable(tf.truncated_normal([vocabulary_size, embeddings_size], stddev=1.0/math.sqrt(embeddings_size)))
softmax_biases=tf.Variable(tf.zeros([vocabulary_size]))

# Model
# lookup embeddings for inputs
embed=tf.nn.embedding_lookup(embeddings, train_dataset)
# compute the softmax loss, using a sample of the negative labels each time
loss=tf.reduce_mean(tf.nn.sampled_softmax_loss(weights=softmax_weights, biases=softmax_biases, inputs=embed, 
                                               labels=train_labels, num_sampled=num_sampled, num_classes=vocabulary_size))
optimzer=tf.train.AdagradOptimizer(1.0).minimize(loss)
# compute the similarity between minibatch examples and all embeddings
norm=tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
normalized_embeddings=embeddings/norm
valid_embeddings=tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
similarity=tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))

num_steps=100001
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for step in range(num_steps):
        batch_data, batch_labels=generate_batch(batch_size, num_skips, skip_window)
        feed_dict={train_dataset: batch_data, train_labels: batch_labels}
        _, l=sess.run([optimzer, loss], feed_dict=feed_dict)
    final_embeddings=normalized_embeddings.eval()
    
# Save data
data={
      'embeddings':final_embeddings, 
      'reverse_dictionary':reverse_dictionary
      }
with open('data.pickle', 'wb') as fw:
    pickle.dump(data, fw, pickle.HIGHEST_PROTOCOL)
with open('data.pickle', 'rb') as fr:
    data_new_embed=pickle.load(fr)
    
# Visualize embeddings
num_points=400
tsne=TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
two_d_embeddings=tsne.fit_transform(final_embeddings[1:num_points+1, :])
def plot(embeddings, labels):
    assert embeddings.shape[0]>=len(labels)
    pylab.figure(figsize=(15, 15))
    for i, label in enumerate(labels):
        x, y=embeddings[i, :]
        pylab.scatter(x, y)
        pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    pylab.show()
words=[reverse_dictionary[i] for i in range(1, num_points+1)]
plot(two_d_embeddings, words)