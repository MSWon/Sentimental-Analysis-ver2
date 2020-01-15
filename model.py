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
import Bi_LSTM

## os.environ['CUDA_VISIBLE_DEVICES'] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


class Model():
    
    def __init__(self,batch_size,word_dim,hidden_dim,num_layers,max_vocab_size,max_word_len,learning_rate,training_epochs,path, isTrain=True):
        
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
                
        ## Placeholders
        self.word_idx = tf.placeholder(tf.int32, shape = [None, max_word_len], name = 'word_idx')
        self.label = tf.placeholder(tf.int32, shape = [None], name = 'label')
        self.seq_len = tf.placeholder(tf.int32, shape = [None], name = 'seq_len')
        self.dropout = tf.placeholder(tf.float32, shape = (), name = 'dropout')
        if isTrain:
            ## Read file
            self.train_text, self.train_len, self.train_score = self.prepro.read_data(self.path + '/ratings_train.txt') 
            self.test_text, self.test_len, self.test_score = self.prepro.read_data(self.path + '/ratings_test.txt') 
            self.train_size, self.test_size = len(self.train_score), len(self.test_score)
            num_train_steps = int(self.train_size / self.batch_size) + 1

            train_dataset = tf.data.Dataset.from_tensor_slices((self.word_idx, self.label, self.seq_len))
            train_dataset = train_dataset.shuffle(self.train_size)
            train_dataset = train_dataset.batch(self.batch_size)
            train_dataset = train_dataset.repeat()

            test_dataset = tf.data.Dataset.from_tensor_slices((self.word_idx, self.label, self.seq_len))
            test_dataset = test_dataset.batch(self.batch_size)
            test_dataset = test_dataset.repeat()

            iters = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
            self.iter_word_idx, self.iter_label, self.iter_seq_len = iters.get_next()

            ## Create the initialisation operations
            self.train_init_op = iters.make_initializer(train_dataset)
            self.test_init_op = iters.make_initializer(test_dataset)

        ## Build graph
        self.build_model(isTrain)
        if isTrain:
            self.build_optimizer(num_train_steps)
            self.get_accuracy()
        
    def train(self):
        num_train_batch = int(self.train_size / self.batch_size) + 1
        num_test_batch = int(self.test_size / self.batch_size) + 1
        
        train_feed_dict = {self.word_idx: self.train_text, self.label: self.train_score,
                           self.seq_len: self.train_len}     
        
        test_feed_dict = {self.word_idx: self.test_text, self.label: self.test_score,
                          self.seq_len: self.test_len}          
        
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

            sess.run(self.train_init_op, feed_dict = train_feed_dict) 
            start_time = datetime.datetime.now()
            
            train_loss_list, test_loss_list = [], []
            train_acc_list, test_acc_list = [], []
            
            for epoch in range(self.training_epochs):
                
                train_acc, train_loss = 0., 0.               
                for step in range(num_train_batch):  
                    loss, acc, _ = sess.run([self.loss, self.accuracy, self.optimizer],
                                                            feed_dict={self.dropout: 0.2})               
                    train_acc += acc/num_train_batch
                    train_loss += loss/num_train_batch
                    print("epoch {:02d} step {:04d} loss {:.6f} accuracy {:.4f}".format(epoch+1, step+1, loss, acc))
                                        
                print("Now for test data\nCould take few minutes")
                sess.run(self.test_init_op, feed_dict = test_feed_dict)
                test_acc, test_loss = 0., 0.              
                for step in range(num_test_batch):                  
                    loss, acc= sess.run([self.loss, self.accuracy], feed_dict={self.dropout: 0.0})
                    test_acc += acc/num_test_batch
                    test_loss += loss/num_test_batch
                
                print("epoch {:02d} [train] loss {:.6f} accuracy {:.4f}".format(epoch+1, train_loss, train_acc))   
                print("epoch {:02d} [test] loss {:.6f} accuracy {:.4f}".format(epoch+1, test_loss, test_acc))
                train_loss_list.append(train_loss)
                train_acc_list.append(train_acc)
                test_loss_list.append(test_loss)
                test_acc_list.append(test_acc)
                sess.run(self.train_init_op, feed_dict = train_feed_dict)
                
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

    def test_example(self):                
        modelpath = "./model/"
        modelName = "sentimental.ckpt"
        saver = tf.train.Saver()  
        
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, modelpath + modelName)
        print("Model loaded!")
        
        pred = tf.cast(tf.argmax(self.logits, 1), tf.int32)
   
        while(1):
            input_words = input("문장을 입력하세요: ")
            tokenize_words = ['/'.join(t) for t in self.prepro.twitter.pos(input_words)]
            test_text = self.prepro.sent2idx(tokenize_words)
            test_len = len(input_words)
            test_feed_dict = {self.word_idx: [test_text],
                              self.seq_len: [test_len],
                              self.dropout:0.0}
            score = sess.run(pred, feed_dict=test_feed_dict)
            if(score == 1):
                print("긍정입니다")
            else:
                print("부정입니다")
            
    def get_accuracy(self):
        pred = tf.cast(tf.argmax(self.logits, 1), tf.int32)
        correct_pred = tf.equal(pred, self.iter_label)
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    def build_embed(self, word_inputs):
        word_emb = tf.nn.embedding_lookup(self.word_embedding, word_inputs)
        return word_emb
   
    def build_model(self, isTrain=True):
        if isTrain:
            word_idx = self.iter_word_idx
            seq_len = self.iter_seq_len
        else:
            word_idx = self.word_idx
            seq_len = self.seq_len
            
        self.word_emb = self.build_embed(word_idx)
        self.model = Bi_LSTM.Bi_LSTM(self.hidden_dim, self.num_layers, self.dropout)
        self.logits = self.model.build_logits(self.word_emb, seq_len)      
               
    def build_optimizer(self, num_train_steps):
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.polynomial_decay(
                  self.learning_rate,
                  global_step,
                  num_train_steps,
                  end_learning_rate=0.0,
                  power=1.0,
                  cycle=False)    
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = self.logits , labels = self.iter_label)) # Softmax loss
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss) # Adam Optimizer            
