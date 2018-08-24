#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Bo Song on 2017/8/8

import math
import multiprocessing
import numpy as np


# Global variables that are shared across processes
_model = None
_testRatings = None
_testNegatives = None
_K = None


def evaluate_model(model, testRatings, testNegatives, K, num_thread):
    """
    Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
    Return: score of each test rating.
    """
    global _model
    global _testRatings
    global _testNegatives
    global _K
    _model = model
    _testRatings = testRatings
    _testNegatives = testNegatives
    _K = K

    hits, ndcgs = [], []
    if (num_thread > 1):  # Multi-thread
        pool = multiprocessing.Pool(processes=num_thread)
        res = pool.map(eval_one_rating, range(len(_testRatings)))
        pool.close()
        pool.join()
        hits = [r[0] for r in res]
        ndcgs = [r[1] for r in res]
        return (hits, ndcgs)
    # Single thread

    for idx in range(len(_testRatings)):
        (hr, ndcg) = eval_one_rating(idx)
        hits.append(hr)
        ndcgs.append(ndcg)
    return (hits, ndcgs)


def eval_one_rating(idx):
    rating = _testRatings[idx]
    items = _testNegatives[idx]
    u = rating[0]
    gtItem = rating[1]
    if gtItem==-1:
        return -1,-1
    # items.append(gtItem)
    # Get prediction scores

    users = np.full(len(items), u, dtype='int32')
    item_pos=np.full(len(items),gtItem,dtype='int32')

    predictions1 = _model.predict([users,item_pos,np.array(items)],batch_size=101, verbose=0)
    predictions2 = _model.predict([users,np.array(items),item_pos], batch_size=101, verbose=0)
    prediction=predictions1-predictions2

    num_err=len(prediction[prediction<0])
    if num_err>=_K:
        hr=0
        ndcg=0
    else:
        hr=1
        ndcg=math.log(2) / math.log(num_err + 2)

    return (hr, ndcg)
















