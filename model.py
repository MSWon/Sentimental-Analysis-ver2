# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 21:19:26 2019

@author: jbk48
"""

import tensorflow as tf
import Bi_LSTM

## os.environ['CUDA_VISIBLE_DEVICES'] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


class Model():
    
    def __init__(self, word_embedding, max_seq_len, word_len=None, word_inputs=None):
        self.word_embedding = word_embedding
        if word_inputs is None and word_len is None:
            self.word_inputs = tf.placeholder(shape=(None, max_seq_len), dtype=tf.int32)
            self.word_len = tf.placeholder(shape=(None), dtype=tf.int32)
        else:
            self.word_inputs = word_inputs
            self.word_len = word_len
            
    def get_accuracy(self, logits, labels):
        pred = tf.cast(tf.argmax(logits, 1), tf.int32)
        correct_pred = tf.equal(pred, labels)
        return tf.reduce_mean(tf.cast(correct_pred, tf.float32))
   
    def build_model(self, hidden_dim, num_layers, seq_len, dropout=None, isTrain=True):
        with tf.variable_scope("Bi-LSTM", reuse=tf.AUTO_REUSE):
            word_emb = tf.nn.embedding_lookup(self.word_embedding, self.word_inputs)
            model = Bi_LSTM.Bi_LSTM(hidden_dim, num_layers, dropout, isTrain)
            logits = model.build_logits(word_emb, self.word_len)
        return logits
    
    def build_loss(self, logits, labels):
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)) # Softmax loss
    
    def build_optimizer(self, loss, learning_rate, num_train_steps):
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.polynomial_decay(
                  learning_rate,
                  global_step,
                  num_train_steps,
                  end_learning_rate=0.0,
                  power=1.0,
                  cycle=False)    
        return tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss) # Adam Optimizer            
        