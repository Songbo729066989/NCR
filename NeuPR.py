#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Bo Song on 2017/8/20
from keras.regularizers import l1,l2
from keras.models import Sequential,Model
from keras.layers import Embedding,Input,Dense,merge,Flatten,Lambda
from keras.optimizers import Adam,Adagrad,SGD,RMSprop,Adadelta
from keras.initializers import RandomNormal,TruncatedNormal
from evaluate_ncr import evaluate_model
from DataSet import DataSet
from time import time
import numpy as np
import argparse
def parse_args():
    dataset='ml1m.txt'
    parser = argparse.ArgumentParser(description="Run NeuPR.")
    parser.add_argument('--path', nargs='?', default='data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default=dataset,
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--num_factors', type=int, default=8,
                        help='Embedding size of MF model.')
    parser.add_argument('--layers', nargs='?', default='[32,32,16,8]',
                        help="MLP layers. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
    parser.add_argument('--reg_mf', type=float, default=0,
                        help='Regularization for MF embeddings.')
    parser.add_argument('--reg_layers', nargs='?', default='[0,0,0,0]',
                        help="Regularization for each MLP layer. reg_layers[0] is the regularization for embeddings.")
    parser.add_argument('--num_neg', type=int, default=2,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    return parser.parse_args()

def get_model(num_users,num_items,mf_dim=10,layers=[10],reg_layers=[0,0,0,0],reg_mf=0):
    num_layer=len(layers)

    user_input=Input(shape=(1,),dtype='int32')
    item_input_pos=Input(shape=(1,),dtype='int32')
    item_input_neg = Input(shape=(1,), dtype='int32')

    MF_embedding_user=Embedding(input_dim=num_users,output_dim=mf_dim,embeddings_initializer='random_normal',
                                name='mf_user_embedding',embeddings_regularizer=l2(reg_mf),input_length=1)
    MF_embedding_item = Embedding(input_dim=num_items, output_dim=mf_dim, embeddings_initializer='random_normal',
                                  name='mf_item_embedding',embeddings_regularizer=l2(reg_mf), input_length=1)
    MLP_embedding_user=Embedding(input_dim=num_users,output_dim=layers[0],embeddings_initializer='random_normal',
                                 name='mlp_user_embedding', embeddings_regularizer=l2(reg_mf),input_length=1)
    MLP_embedding_item = Embedding(input_dim=num_items, output_dim=layers[0], embeddings_initializer='random_normal',
                                   name='mlp_item_embedding',embeddings_regularizer=l2(reg_mf), input_length=1)

    mf_user_latent=Flatten()(MF_embedding_user(user_input))
    mf_item_latent_pos=Flatten()(MF_embedding_item(item_input_pos))
    mf_item_latent_neg = Flatten()(MF_embedding_item(item_input_neg))


    prefer_pos = merge([mf_user_latent, mf_item_latent_pos], mode='mul')
    prefer_neg = merge([mf_user_latent, mf_item_latent_neg], mode='mul')
    prefer_neg = Lambda(lambda x: -x)(prefer_neg)
    mf_vector = merge([prefer_pos, prefer_neg], mode='concat')


    mlp_user_latent=Flatten()(MLP_embedding_user(user_input))
    mlp_item_latent_pos=Flatten()(MLP_embedding_item(item_input_pos))
    mlp_item_latent_neg=Flatten()(MLP_embedding_item(item_input_neg))
    mlp_item_latent_neg=Lambda(lambda x:-x)(mlp_item_latent_neg)
    mlp_vector=merge([mlp_user_latent,mlp_item_latent_pos,mlp_item_latent_neg],mode='concat')
    for idx in range(1,num_layer):
        layer=Dense(layers[idx],kernel_regularizer=l2(0.0000),activation='tanh',name="layer%d" %idx)
        mlp_vector=layer(mlp_vector)

    predict_vector=merge([mf_vector,mlp_vector],mode='concat')

    prediction=Dense(1,activation='sigmoid',kernel_initializer='lecun_uniform',name='prediction')(predict_vector)
    model=Model(inputs=[user_input,item_input_pos,item_input_neg],outputs=prediction)

    return model

def get_train_instances(train, num_negatives):
    user_input,item_pos,item_neg,labels = [],[],[],[]
    num_items= train.shape[1]
    for (u, i) in train.keys():
        # positive instance
        user_input.append(u)
        item_pos.append(i)
        j = np.random.randint(num_items)
        while (u, j) in train.keys():
            j = np.random.randint(num_items)
        item_neg.append(j)
        labels.append(1)

        user_input.append(u)
        item_pos.append(j)
        item_neg.append(i)
        labels.append(0)

        # negative instances

        for cnt in range(num_negatives-1):
            user_input.append(u)
            j=np.random.randint(num_items)
            while (u,j) in train.keys():
                j = np.random.randint(num_items)
            item_pos.append(j)
            item_neg.append(i)
            labels.append(0)

    # np.random.seed(123)
    # np.random.shuffle(user_input)
    # np.random.seed(123)
    # np.random.shuffle(item_pos)
    # np.random.seed(123)
    # np.random.shuffle(item_neg)
    # np.random.seed(123)
    # np.random.shuffle(labels)
    return user_input, item_pos,item_neg,labels

if __name__=='__main__':
    args = parse_args()
    num_epochs = args.epochs
    batch_size = args.batch_size
    mf_dim = args.num_factors
    layers = eval(args.layers)
    reg_mf = args.reg_mf
    reg_layers = eval(args.reg_layers)
    num_negatives = args.num_neg
    # learning_rate = args.lr
    learner = args.learner
    verbose = args.verbose

    topK = 10
    evaluation_threads = 1#mp.cpu_count()
    print("NeuPR arguments: %s " %(args))
    num_epochs=50

    num_negatives=2
    learning_rate = 0.0005

    k=2 # k=[1,2,3,4,8]
    layers=[e*k for e in layers]

    learner='adam'
    t1 = time()
    dataset = DataSet()
    dataset.loadClicks('data/' + args.dataset, 10, 10)
    train,validRatings, testRatings, testNegatives = dataset.trainMatrix,dataset.validRatings,dataset.testRatings, dataset.testNegatives
    num_users, num_items = train.shape
    print(train.shape)
    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d"
          % (time() - t1, num_users, num_items, train.nnz, len(testRatings)))

    ######################################################
    model = get_model(num_users, num_items, mf_dim, layers, reg_layers, reg_mf)

    if learner.lower() == "adagrad":
        model.compile(optimizer=Adagrad(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "rmsprop":
        model.compile(optimizer=RMSprop(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "adam":
        model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy')
    else:
        model.compile(optimizer=SGD(lr=learning_rate), loss='binary_crossentropy')
        print('sgd')

    (hits, ndcgs) = evaluate_model(model, validRatings, testNegatives, topK, evaluation_threads)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print('Init: HR = %.4f, NDCG = %.4f' % (hr, ndcg))
    best_hr, best_ndcg, best_iter = hr, ndcg, -1
    best_test_hr,best_test_ndcg=-1,-1
        # Training model
    all_loss = []
    for epoch in range(num_epochs):
        t1 = time()
        # Generate training instances
        user_input, item_input_pos,item_input_neg, labels = get_train_instances(train, num_negatives)

        # Training
        hist = model.fit([np.array(user_input), np.array(item_input_pos),np.array(item_input_neg)],  # input
                         np.array(labels),  # labels
                         batch_size=batch_size, epochs=1, verbose=0, shuffle=True)
        t2 = time()

        # model_out_file = 'Pretrain/%s_NeuPR_%d_%s_%d.h5' % (args.dataset, mf_dim, args.layers, epoch)
        # model_out_file = 'Pretrain/%s_NeuPR_%d.h5' % (args.dataset, mf_dim)

        if epoch % verbose == 0:
            (hits, ndcgs) = evaluate_model(model, validRatings, testNegatives, topK, evaluation_threads)
            hits,ndcgs=np.array(hits),np.array(ndcgs)
            vhr, vndcg=hits[hits>=0].mean(), ndcgs[ndcgs>=0].mean()
            (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
            hits, ndcgs = np.array(hits), np.array(ndcgs)
            hr, ndcg, loss = hits[hits>=0].mean(), ndcgs[ndcgs>=0].mean(), hist.history['loss'][0]
            all_loss.append(loss)
            print('Iteration %d [%.1f s]: [Valid HR = %.4f, NDCG = %.4f, loss=%.6f]\t[Test HR = %.4f, NDCG = %.4f], [%.1f s]'
                  % (epoch, t2 - t1, vhr,vndcg,loss,hr,ndcg ,time() - t2))
            # print('Iteration %d [%.1f s]: Test HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]'
            #       % (epoch, t2 - t1, hr, ndcg, loss, time() - t2))
            if vhr > best_hr:
                best_hr, best_ndcg, best_iter = vhr, vndcg, epoch
                best_test_hr,best_test_ndcg=hr,ndcg

    print("End. Best Iteration %d: Test HR = %.4f, NDCG = %.4f. " % (best_iter, best_test_hr, best_test_ndcg))
    print('learning_rate: %.5f , num_factor: %d' % (learning_rate, mf_dim))






