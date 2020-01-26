# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 21:19:26 2019

@author: jbk48
"""
import os
import datetime
import pandas as pd
import tensorflow as tf
import preprocess
from model import Model

## os.environ['CUDA_VISIBLE_DEVICES'] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


class Train():
    
    def __init__(self,batch_size,word_dim,hidden_dim,num_layers,max_vocab_size,max_word_len,learning_rate,training_epochs,path):
        
        self.batch_size = batch_size
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_vocab_size = max_vocab_size
        self.max_word_len = max_word_len
        self.learning_rate = learning_rate
        self.training_epochs = training_epochs
        self.path = path

        ## Preprocess data
        self.prepro = preprocess.Preprocess(word_dim=word_dim, max_vocab_size=max_vocab_size, path=path)
        self.word_embedding, self.clear_padding, self.word2idx, self.idx2word = self.prepro.build_embedding()

        ## Read file
        self.train_text, self.train_len, self.train_score = self.prepro.read_data(self.path + '/ratings_train.txt') 
        self.test_text, self.test_len, self.test_score = self.prepro.read_data(self.path + '/ratings_test.txt') 
        self.train_size, self.test_size = len(self.train_score), len(self.test_score)
        self.num_train_steps = int(self.train_size / self.batch_size) + 1

        train_dataset = tf.data.Dataset.from_tensor_slices((self.train_text, self.train_len, self.train_score))
        train_dataset = train_dataset.shuffle(self.train_size)
        train_dataset = train_dataset.batch(self.batch_size)
        train_dataset = train_dataset.repeat()

        test_dataset = tf.data.Dataset.from_tensor_slices((self.test_text, self.test_len, self.test_score))
        test_dataset = test_dataset.batch(self.batch_size)
        test_dataset = test_dataset.repeat()

        iters = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
        self.iter_word_idx, self.iter_seq_len, self.iter_label = iters.get_next()

        ## Create the initialisation operations
        self.train_init_op = iters.make_initializer(train_dataset)
        self.test_init_op = iters.make_initializer(test_dataset)

        ## Build graph
        Train_graph = Model(self.word_embedding, self.max_word_len, self.iter_seq_len, self.iter_word_idx)
        Test_graph = Model(self.word_embedding, self.max_word_len, self.iter_seq_len, self.iter_word_idx)
        self.train_fn = Train_graph.build_model(self.hidden_dim, self.num_layers, self.iter_seq_len, 0.2, isTrain=True)
        self.test_fn = Test_graph.build_model(self.hidden_dim, self.num_layers, self.iter_seq_len, None, isTrain=False)
        self.train_loss = Train_graph.build_loss(self.train_fn, self.iter_label)
        self.train_acc = Train_graph.get_accuracy(self.train_fn, self.iter_label)
        self.train_op = Train_graph.build_optimizer(self.train_loss, self.learning_rate, self.num_train_steps)
        self.test_loss = Test_graph.build_loss(self.test_fn, self.iter_label)
        self.test_acc = Test_graph.get_accuracy(self.test_fn, self.iter_label)
        
    def train(self):
        num_train_batch = int(self.train_size / self.batch_size) + 1
        num_test_batch = int(self.test_size / self.batch_size) + 1
                    
        print("start training")
        modelpath = "./model/"
        modelName = "sentimental.ckpt"
        saver = tf.train.Saver()  
        best_acc = 0.
        
        with tf.Session(config=config) as sess:
            
            sess.run(tf.global_variables_initializer())

            if(not os.path.exists(modelpath)):
                os.mkdir(modelpath)
            ckpt = tf.train.get_checkpoint_state(modelpath)
            
            if(ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path)):
                saver.restore(sess, modelpath + modelName)
                print("Model loaded!")

            start_time = datetime.datetime.now()
            
            train_loss_list, test_loss_list = [], []
            train_acc_list, test_acc_list = [], []
            
            for epoch in range(self.training_epochs):
                sess.run(self.train_init_op)    
                train_acc, train_loss = 0., 0.               
                for step in range(num_train_batch):  
                    loss, acc, _ = sess.run([self.train_loss, self.train_acc, self.train_op])               
                    train_acc += acc/num_train_batch
                    train_loss += loss/num_train_batch
                    print("epoch {:02d} step {:04d} loss {:.6f} accuracy {:.4f}".format(epoch+1, step+1, loss, acc))
                                        
                print("Now for test data\nCould take few minutes")
                sess.run(self.test_init_op)
                test_acc, test_loss = 0., 0.              
                for step in range(num_test_batch):                  
                    loss, acc= sess.run([self.test_loss, self.test_acc])
                    test_acc += acc/num_test_batch
                    test_loss += loss/num_test_batch
                
                print("epoch {:02d} [train] loss {:.6f} accuracy {:.4f}".format(epoch+1, train_loss, train_acc))   
                print("epoch {:02d} [test] loss {:.6f} accuracy {:.4f}".format(epoch+1, test_loss, test_acc))
                train_loss_list.append(train_loss)
                train_acc_list.append(train_acc)
                test_loss_list.append(test_loss)
                test_acc_list.append(test_acc)
                
                if(best_acc <= test_acc):
                    best_acc = test_acc
                    save_path = saver.save(sess, modelpath + modelName)
                    print ('save_path',save_path)
                
            result = pd.DataFrame({"train_loss":train_loss_list,
                                   "train_accuracy":train_acc_list,
                                   "test_loss":test_loss_list,
                                   "test_accuracy":test_acc_list})
            
            result.to_csv("./loss.csv", sep =",", index=False)
            elapsed_time = datetime.datetime.now() - start_time
            print("{}".format(elapsed_time))

        
        