# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 22:10:30 2019

@author: jbk48
"""

from train import Train
import tensorflow as tf

if __name__ == '__main__':

    flags = tf.app.flags
    FLAGS = flags.FLAGS
    
    ## Model parameter
    flags.DEFINE_integer('batch_size', 128, 'number of batch size')
    flags.DEFINE_integer('word_dim', 512, 'dimension of word vector')
    flags.DEFINE_integer('hidden_dim', 512, 'dimension of model hidden nodes')
    flags.DEFINE_integer('num_layers', 2, 'number of LSTM layers')
    flags.DEFINE_integer('max_vocab_size', 100000, 'max vocabulary size')
    flags.DEFINE_integer('max_word_len', 100, 'max length of words')
    flags.DEFINE_float('learning_rate', 0.0001, 'initial learning rate')
    flags.DEFINE_string('path', 'Movie_rating_data', 'directory of data file')
    flags.DEFINE_integer('training_epochs', 5, 'number of training epochs')
    
    print('========================')
    for key in FLAGS.__flags.keys():
        print('{} : {}'.format(key, getattr(FLAGS, key)))
    print('========================')
    ## Build model
    train_fn = Train(FLAGS.batch_size, FLAGS.word_dim, FLAGS.hidden_dim, FLAGS.num_layers, FLAGS.max_vocab_size,
                     FLAGS.max_word_len, FLAGS.learning_rate, FLAGS.training_epochs, FLAGS.path)
    
    ## Train model
    train_fn.train()
