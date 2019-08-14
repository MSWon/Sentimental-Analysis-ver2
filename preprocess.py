# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 18:53:34 2019

@author: jbk48
"""
import os
import tensorflow as tf
import pickle
from konlpy.tag import Twitter

class Preprocess():
    
    def __init__(self, word_dim=512, max_len=100, max_vocab_size=100000, path="./Movie_rating_data"):
        
        self.word_dim = word_dim
        self.max_len = max_len
        self.max_vocab_size = max_vocab_size
        self.path = path
        self.twitter = Twitter()
        self.build_vocab()
        
    def read_data(self, filename):
        if(os.path.exists("./{}.pkl".format(filename))):
            print("getting {}".format(filename))
            with open("./{}.pkl".format(filename), "rb") as f:
                data = pickle.load(f)
        else:
            print("reading {} file".format(filename))
            print("could take a while")
            text, seq_len, score = [], [], []
            with open(filename, 'r',encoding='utf-8') as f:
                for line in f.read().splitlines():
                    data = line.split('\t')
                    pos_tag = ['/'.join(t) for t in self.twitter.pos(data[1])]
                    text.append(self.sent2idx(pos_tag))
                    seq_len.append(len(pos_tag))
                    score.append(data[2])
            print("size : {}".format(len(text[1:])))
            data = text[1:], seq_len[1:], list(map(int,score[1:])) ## Remove header
            with open("./{}.pkl".format(filename), "wb") as f:
                pickle.dump(data, f)
                
        return data 
    
    def sent2idx(self, sent):
        idx = []
        for word in sent:
            if word not in self.word2idx:
                idx.append(self.word2idx["<UNK>"])
            else:
                idx.append(self.word2idx[word])
        idx = idx + [0]*(self.max_len-len(idx)) ## PAD
        return idx
    
    def build_vocab(self):
        if(os.path.exists("word2idx.pkl")):
            print("getting vocab")
            with open('./word2idx.pkl', 'rb') as f:
                self.word2idx = pickle.load(f) 
            self.idx2word = dict(zip(self.word2idx.values(), self.word2idx.keys()))
            self.vocab_size = len(self.word2idx)
        else:
            print("building vocab")
            print("could take a while")
            word_vocab = {}
            with open(self.path + '/ratings_train.txt', 'r',encoding='utf-8') as f:
                for line in f.read().splitlines():
                    data = line.split('\t')
                    text = ['/'.join(t) for t in self.twitter.pos(data[1])]
               
                    for word in text:
                        if(word not in word_vocab):
                            word_vocab[word] = 1
                        elif(word in word_vocab):
                            word_vocab[word] += 1
                        
            word_freq = sorted(word_vocab.items(), key = lambda x : x[1], reverse = True)
            word_freq_ = word_freq[:self.max_vocab_size] ## Top freq vocabs (100K for default)   
            
            self.word2idx = {"<PAD>":0, "<UNK>":1} ## predict as <UNK> for OOV word            
            for idx in range(len(word_freq_)):
                self.word2idx[word_freq_[idx][0]] = len(self.word2idx)
            
            self.idx2word = dict(zip(self.word2idx.values(), self.word2idx.keys()))
            self.vocab_size = len(self.word2idx)
            
            with open('./word2idx.pkl', 'wb') as f:
                pickle.dump(self.word2idx, f)
        
    def build_embedding(self):
        print("building word embedding")
        with tf.variable_scope('word-embedding', reuse = False):
            self.word_embedding = tf.get_variable(name="word-emb", shape=[self.vocab_size, self.word_dim],
                                                  dtype=tf.float32, initializer = tf.contrib.layers.xavier_initializer())  
        self.clear_padding = tf.scatter_update(self.word_embedding, [0], tf.constant(0.0, shape=[1, self.word_dim])) 
        return self.word_embedding, self.clear_padding, self.word2idx, self.idx2word
        


