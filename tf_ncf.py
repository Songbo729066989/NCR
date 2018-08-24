#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Bo Song on 2018/7/7

import tensorflow as tf
import numpy as np
from time import time
from evaluate_ncf import evaluate_model
tf.set_random_seed(2018)

class NCF:
    def __init__(self, config, num_user,num_item):
        self.config = config
        self.num_users = num_user
        self.num_items = num_item

        self.dim_emb = config['dim_emb']

        self.batch_size = config['batch_size']
        self.lr=config['learning_rate']
        self.istraining = True
        self.build_model()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())


    def build_model(self):

        self.u = tf.placeholder(tf.int32, shape=[None, ])
        self.i = tf.placeholder(tf.int32, shape=[None, ])
        self.y = tf.placeholder(tf.float32, shape=[None,],name="labels")
        #
        self.user_embeds = tf.Variable(tf.random_normal([self.num_users, self.dim_emb],mean=0.0,stddev=0.01), name='user_embs')
        self.item_embeds = tf.Variable(tf.random_normal([self.num_items, self.dim_emb],mean=0.0,stddev=0.01), name='item_embs')

        # self.user_embeds = tf.keras.layers.Embedding(input_dim = self.num_users, output_dim = self.dim_emb, name = 'user_embedding',
        #                           embeddings_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=2018), embeddings_regularizer=tf.keras.regularizers.l2(0), input_length=1)
        # self.item_embeds = tf.keras.layers.Embedding(input_dim=self.num_items, output_dim=self.dim_emb,
        #                                              name='item_embedding',
        #                                              embeddings_initializer=tf.keras.initializers.RandomNormal(mean=0.0,
        #                                                                                                        stddev=0.05,
        #                                                                                                        seed=2018),
        #                                              embeddings_regularizer=tf.keras.regularizers.l2(0), input_length=1)

        self.compute_loss()

        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)






    def compute_loss(self):


        self.static_u_emb = tf.nn.embedding_lookup(self.user_embeds,self.u)
        self.static_i_emb = tf.nn.embedding_lookup(self.item_embeds,self.i)

        p1 = tf.multiply(self.static_u_emb,self.static_i_emb)

        # self.out = tf.layers.dense(p1,units=1,activation=tf.nn.sigmoid)
        # self.out  = tf.nn.sigmoid(tf.add(tf.matmul(p1,output_w),output_b))
        # self.out=tf.squeeze(self.out)
        # # self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y,logits=self.out))
        # self.loss = tf.losses.log_loss(labels=self.y,predictions=self.out)


        p2 = tf.concat([self.static_u_emb,self.static_i_emb],axis=1)
        p2 = tf.layers.dense(p2,units=2*self.dim_emb,activation=tf.nn.relu)
        p2 = tf.layers.dense(p2,units=self.dim_emb,activation=tf.nn.relu)
        p2 = tf.layers.dense(p2,units=int(0.5*self.dim_emb),activation=tf.nn.relu)

        p_final = tf.concat([p1,p2],axis=1)
        self.out = tf.layers.dense(p_final,units=1,activation=tf.nn.sigmoid)
        self.out = tf.squeeze(self.out)
        # self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y,logits=self.out))
        self.loss = tf.losses.log_loss(labels=self.y, predictions=self.out)



def get_train_instances(train, num_negatives):
    user_input,item_input,labels = [],[],[]
    num_items= train.shape[1]

    for (u, i) in train.keys():
        # positive instance
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        # negative instance
        for _ in range(num_negatives):
            j = np.random.randint(num_items)
            while (u, j) in train.keys():
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)

    return user_input, item_input,labels




if __name__=='__main__':
    from DataSet import DataSet

    dataset = DataSet()
    dataset.loadClicks('data/'+'ml1m.txt',10,10)
    train,validRatings,testRatings,testNegatives  = dataset.trainMatrix,dataset.validRatings,dataset.testRatings,dataset.testNegatives
    num_user,num_item = train.shape


    config = {
        'dim_emb': 32,
        'batch_size': 256,
        'learning_rate': 0.001,
        'l2_reg': 0.0001
    }

    model = NCF(config,num_user,num_item)

    sess = model.sess

    print('training start.....')
    topK =10

    for epoch in range(30):

        user_input, item_input, labels = get_train_instances(train, 4)

        user_input = np.array(user_input, dtype=np.int32)
        item_input = np.array(item_input, dtype=np.int32)
        labels = np.array(labels, dtype=np.float32)

        num_training = len(user_input)

        np.random.seed(0)

        shuffle_index = np.arange(len(user_input))
        np.random.shuffle(shuffle_index)
        user_input=user_input[shuffle_index]
        item_input=item_input[shuffle_index]
        labels = labels[shuffle_index]

        total_loss = 0.0

        st_time = time()

        for num_batch in range(int(num_training/config['batch_size'])):
            id_start = num_batch * config['batch_size']
            id_end = (num_batch + 1) * config['batch_size']
            if id_end>num_training:
                id_end=num_training

            bat_u = user_input[id_start:id_end]
            bat_i = item_input[id_start:id_end]
            bat_la = labels[id_start:id_end]


            loss,_ = model.sess.run((model.loss,model.train_op),
                                    feed_dict={
                                        model.u:bat_u,
                                        model.i:bat_i,
                                        model.y:bat_la,
                                    })
            total_loss+=loss*(id_end-id_start)
        print("[iter %d : loss : %f, time: %f]" %(epoch,total_loss/num_training,time()-st_time))

        t1=time()
        (hits, ndcgs) = evaluate_model(model, validRatings, testNegatives, topK, 1)
        hits, ndcgs = np.array(hits), np.array(ndcgs)
        vhr, vndcg = hits[hits >= 0].mean(), ndcgs[ndcgs >= 0].mean()
        (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, 1)
        hits, ndcgs = np.array(hits), np.array(ndcgs)
        hr, ndcg = hits[hits >= 0].mean(), ndcgs[ndcgs >= 0].mean()
        print('Iteration %d [%.1f s]: [Valid HR = %.4f, NDCG = %.4f]\t[Test HR = %.4f, NDCG = %.4f]'
            % (epoch, time() - t1, vhr, vndcg, hr, ndcg))


