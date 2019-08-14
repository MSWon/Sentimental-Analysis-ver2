# -*- coding: utf-8 -*-
"""
Created on Mon May 21 14:30:52 2018

@author: jbk48
"""

import tensorflow as tf

class Bi_LSTM():
    
    def __init__(self, lstm_units, num_layer, dropout):
        
        self.lstm_units = lstm_units
        self.num_layer = num_layer
        self.dropout = dropout
        
        with tf.variable_scope('output-layer', reuse = False):           
            self.W = tf.get_variable(name="W", shape=[2 * lstm_units, 2],
                                dtype=tf.float32, initializer = tf.contrib.layers.xavier_initializer())
            self.b = tf.get_variable(name="b", shape=[2], dtype=tf.float32,
                                initializer=tf.zeros_initializer())
    
    def build_logits(self, word_emb, seq_len):
        output = tf.identity(word_emb)
        for layer in range(1, self.num_layer+1):
            with tf.variable_scope('encoder_{}'.format(layer), reuse=False):
                
                with tf.variable_scope('forward', reuse = False):            
                    self.lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(self.lstm_units, forget_bias=1.0, state_is_tuple=True)
                    self.lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(self.lstm_fw_cell, output_keep_prob = 1. - self.dropout)
                
                with tf.variable_scope('backward', reuse = False):            
                    self.lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(self.lstm_units, forget_bias=1.0, state_is_tuple=True)
                    self.lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(self.lstm_bw_cell, output_keep_prob = 1. - self.dropout)
                
                outputs, states = tf.nn.bidirectional_dynamic_rnn(self.lstm_fw_cell, 
                                                                  self.lstm_bw_cell,
                                                                  dtype=tf.float32,
                                                                  inputs=output, 
                                                                  sequence_length=seq_len)
                output = tf.concat(outputs, 2)
                
        ## concat fw, bw final states
        outputs = tf.concat([states[0][1], states[1][1]], axis=1)
        logits = tf.matmul(outputs, self.W) + self.b        
        return logits
        
    
