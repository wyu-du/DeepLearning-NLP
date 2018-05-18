# -*- coding: utf-8 -*-
"""
Created on Fri May 18 15:04:47 2018

@author: Administrator
"""

import numpy as np 
import nltk

class cbow:
    
    def __init__(self, text, window=2, n=2, learning_rate=0.0001):
        self.n=n
        self.text=text
        self.window=window
        self.learning_rate=learning_rate
        self.vocabulary=list(set(text.lower().split()))
        self.word_index={k:v for v, k in enumerate(self.vocabulary)}
        self.V=np.random.uniform(-0.001, 0.001, (n, len(self.vocabulary)))
        self.U=np.random.uniform(-0.001, 0.001, (len(self.vocabulary), n))
        print('word vector size=', self.n)
        print('vocabulary size=', self.vocabulary)
        print('word index dict='. self.word_index)
        print('shape of V='+str(self.V.shape))
        print('shape of U='+str(self.U.shape))
        
    def get_one_hot(self, index):
        one_hot=np.zeros(len(self.vocabulary))
        one_hot[index]=1
        return one_hot
    
    def get_representation_from_onehot(self, one_hot=None):
        tmp=np.dot(self.V, one_hot)
        return tmp
    
    def get_average_vector_context(self, left_context=None, right_context=None):
        '''
        Given the words in the left and right context, 
        generate the average vector for the context of the current word.
        '''
        avg_vector=np.zeros(self.n)
        
        for word in left_context:
            current_word_index=self.word_index[word]
            one_hot_encoding=self.get_one_hot(current_word_index)
            avg_vector+=self.get_representation_from_onehot(one_hot=one_hot_encoding)
            
        for word in right_context:
            current_word_index=self.word_index[word]
            one_hot_encoding=self.get_one_hot(current_word_index)
            avg_vector+=self.get_representation_from_onehot(one_hot=one_hot_encoding)
            
        avg_vector=avg_vector/(2*self.window)
        return avg_vector
    
    def get_score(self, avg_vec=None):
        '''
        Given an averaged vector in the current context of the center word, 
        compute the product of the U with avg_vec.
        '''
        print('U shape='+str(self.U.shape))
        print('Average vector shape='+str(avg_vec.shape))
        return np.dot(self.U, avg_vec)
    
    def softmax(self, x):
        if x.ndim>1:
            x -= np.max(x, axis=-1).reshape(-1, 1)
            x = np.exp(x)/np.sum(np.exp(x), axis=-1).reshape(-1, 1)
        else:
            x -= np.max(x)
            x = np.exp(x)/np.sum(np.exp(x))
        return x
    
    def compute_cross_entropy_error(self, y, y_hat):
        """
		 Given One hot encoding and the output of the softmax function, 
        this function computes the cross entropy error. 
		 
        Parameters 
		 ---------
		 y : one_hot encoding of the current center word 
		 y_hat : output of the softmax function. 
		 """
        return -y*np.log(y_hat)
    
    def compute_EH(self, error):
        '''
        Given error and U matrix, compute the value of EH
        '''
        EH=np.zeros(self.n)
        for i in range(self.n):
            temp=0.0
            for j in range(len(self.vocabulary)):
                temp+=error[j]*self.U[i,j]
            EH[i]=temp
        return EH
    
    def update_U(self, error, avg_vec):
        for i in range(self.U.shape[0]):
            self.U[i,:] -= self.learning_rate*error[i]*avg_vec
            
    def update_V(self, error, left_context, right_context):
        # compute EH term 
        EH=self.compute_EH(error)
        
        # pass the update
        for word in left_context+right_context:
            current_word_index=self.word_index[word]
            self.V[:, current_word_index] -= (2*self.window)*self.learning_rate*EH
            
    def fit(self):
        self.split_text=self.text.lower().split()
        for i in range(len(self.split_text)):
            print('\n\n')
            center_word=self.split_text[i]
            if i-self.window<0:
                left_window, right_window = self.split_text[:i], self.split_text[i+1, i+self.window]
            else:
                left_window, right_window = self.split_text[i-self.window:i], self.split_text[i+1, i+self.window]
            print('Center word = %s, left_window = %s, right_window = %s' %(center_word, left_window, right_window))
            
            avg_vector=self.get_average_vector_context(left_context=left_window, right_context=right_window)
            
            x_hat=self.get_score(avg_vector)
            print('x_hat shape = '+str(x_hat.shape))
            
            y_hat=self.softmax(x_hat)
            
            center_word_index=self.word_index[center_word]
            center_word_one_hot_encoding=self.get_one_hot(center_word_index)
            
            error=self.compute_cross_entropy_error(center_word_one_hot_encoding, y_hat)
            print('Error = %s', error)
            
            self.update_U(error=error, avg_vec=avg_vector)
            self.update_V(error=error, left_context=left_window, right_context=right_window)
        
        print('U matrix = %s'+str(self.U))
        print('V matrix = %s'+str(self.V))
        

text=nltk.corpus.gutenberg.words('austen-emma.txt')
text = ' '.join(text[:100])

model  = cbow(text=text)
model.fit()

final_vectors = {}
for k, v in model.word_index.items():
	final_vectors[k] = (model.U[v, :] + model.V[:, v])/2
print(final_vectors)

import matplotlib.pylab as plt
for k, v in final_vectors.items():
	plt.text(v[0], v[1], k)
plt.show()