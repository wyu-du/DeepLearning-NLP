# -*- coding: utf-8 -*-
"""
Created on Mon May 28 13:56:24 2018

@author: Administrator
"""

import numpy as np

from keras.engine import Layer, InputSpec
from keras import backend as K
from keras import initializers
from keras import regularizers
from keras import constraints


def dot_product(x, kernel):
    '''
    Args:
        x(): input
        kernel(): weights
    '''
    if K.backend()=='tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)
    

class AttentionWithContext(Layer):
    '''
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    '''
    def __init__(self, W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None, bias=True, **kwargs):
        self.init=initializers.get('glorot_uniform')
        
        self.W_regularizer=regularizers.get(W_regularizer)
        self.u_regularizer=regularizers.get(u_regularizer)
        self.b_regularizer=regularizers.get(b_regularizer)
        
        self.W_constraint=constraints.get(W_constraint)
        self.u_constraint=constraints.get(u_constraint)
        self.b_constraint=constraints.get(b_constraint)
        
        self.bias=bias
        super(AttentionWithContext, self).__init__(**kwargs)
        
    def build(self, input_shape):
        assert len(input_shape)==3
        
        self.W=self.add_weight((input_shape[-1], input_shape[-1], ),
                               initializer=self.init,
                               name='{}_W'.format(self.name),
                               regularizer=self.W_regularizer,
                               constraint=self.W_constraint)
        
        if self.bias:
            self.b=self.add_weight((input_shape[-1], ), 
                                   initializer='zero', 
                                   name='{}_b'.format(self.name),
                                   regularizer=self.b_regularizer,
                                   constraint=self.b_constraint)
        
        self.u=self.add_weight((input_shape[-1], ),
                               initializer=self.init,
                               name='{}_u'.format(self.name),
                               regularizer=self.u_regularizer,
                               constraint=self.u_constraint)
        
        super(AttentionWithContext, self).build(input_shape)
        
    def call(self, x):
        uit=dot_product(x, self.W)
        if self.bias:
            uit+=self.b
        uit=K.tanh(uit)
        ait=dot_product(uit, self.u)
        
        a=K.softmax(ait)
        a=K.expand_dims(a)
        weighted_input=x*a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]        


def visualize_attention(test_seq, model, id2wrd, n):
    get_layer_output=K.function([model.layers[0].input, K.learning_phase()], [model.layers[4].output])
    out=get_layer_output([test_seq, ])(0)
    
    attention_w=model.layers[5].get_weights()
    eij=np.tanh(np.dot(out[0], attention_w[0]))
    ai=np.exp(eij)
    weights=ai/np.sum(ai)
    weights=np.sum(weights, axis=1)
    topkeys=np.argpartition(weights, -n)[-n:]
    print(' '.join([id2wrd[wrd_id] for wrd_id in test_seq[0] if wrd_id!=0]))
    
    for k in test_seq[0][topkeys]:
        print(id2wrd[k])
    