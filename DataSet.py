#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Bo Song on 2017/8/8
import scipy.sparse as sp
import numpy as np

# class DataSet:
#
#     def __init__(self):
#         self.pos_per_user = None
#         self.nUsers = 0
#         self.nItems = 0
#         self.nClicks = 0
#
#         self.userids = {}
#         self.itemids = {}
#
#         self.rUserids = {}
#         self.rItemids={}
#
#         self.pos_per_user={}
#
#     # def loadData(self, filepath, userMin, itemMin):
#
#     def loadClicks(self, filepath, userMin, itemMin):
#
#         uCounts={}
#         iCounts={}
#         nRead=0
#         print("  Loading clicks from %s, userMin = %d  itemMin = %d ", filepath, userMin, itemMin)
#
#         with open(filepath, 'r') as f:
#             for line in f.readlines():
#                 uName, iName, rating, time= line.strip().split(' ')
#                 nRead+=1
#                 if uName not in uCounts:
#                     uCounts[uName]=0
#                 if iName not in iCounts:
#                     iCounts[iName]=0
#
#                 uCounts[uName]+=1
#                 iCounts[iName]+=1
#         print("\n  First pass: #users = %d, #items = %d, #clicks = %d\n",
#         len(uCounts),len(iCounts), nRead)
#
#
#
#         with open(filepath, 'r') as f:
#             for line in f.readlines():
#                 uName, iName, rating, time= line.strip().split(' ')
#                 try:
#                     tmp = int(time)
#                 except:
#                     continue
#
#                 if uCounts[uName]< userMin:
#                     continue
#
#                 if iCounts[iName]< itemMin:
#                     continue
#
#                 self.nClicks+=1
#
#                 if  iName not in self.itemids:
#                     self.rItemids[self.nItems]=iName
#                     self.itemids[iName]=self.nItems
#                     self.nItems+=1
#
#                 if uName not in self.userids:
#                     self.rUserids[self.nUsers]=uName
#                     self.userids[uName]=self.nUsers
#                     self.nUsers+=1
#                     self.pos_per_user[self.userids[uName]]=[]
#                 self.pos_per_user[self.userids[uName]].append((self.itemids[iName], int(time)))
#
#             print("  Sorting clicks for each users ")
#
#             for u in range(self.nUsers):
#                 sorted(self.pos_per_user[u], key=lambda d: d[1])
#             print("\n \"nUsers\": %d,\"nItems\":%d, \"nClicks\":%d\n",self.nUsers,self.nItems,self.nClicks)
#
#             self.val_per_user = []
#             self.test_per_user = []
#             self.train_per_user={}
#             self.test_negative_per_user = {}
#             mat = sp.dok_matrix((self.nUsers, self.nItems), dtype=np.float32)
#
#             # np.random.seed(2017)
#             for u in range(self.nUsers):
#                 if len(self.pos_per_user[u])<2:
#                     # print("Oops! dataset error")
#                     # exit(1)
#                     continue
#                 item_test=self.pos_per_user[u][-1][0]
#                 self.pos_per_user[u].pop()
#
#                 # item_val=self.pos_per_user[u][-1][0]
#                 # self.pos_per_user[u].pop()
#                 self.train_per_user[u]=[]
#                 self.train_per_user[u].append([e[0] for e in self.pos_per_user[u]])
#
#                 self.test_per_user.append([u,item_test])
#                 # self.val_per_user.append([u,item_val])
#
#                 for item in self.train_per_user[u]:
#                     mat[u,item]=1.0
#
#                 self.test_negative_per_user[u]=[]
#                 for i in range(100):
#                     neg_item_id = np.random.randint(0,self.nItems)
#                     while neg_item_id in self.train_per_user[u] or neg_item_id==item_test \
#                           or neg_item_id in self.test_negative_per_user[u]:
#                         neg_item_id = np.random.randint(0, self.nItems)
#                     self.test_negative_per_user[u].append(neg_item_id)
#             self.trainMatrix = mat
#             self.testRatings = self.test_per_user
#             self.testNegatives=[]
#             for u in range(self.nUsers):
#                 if u in self.train_per_user:
#                     self.testNegatives.append([e for e in self.test_negative_per_user[u]])


class DataSet:

    def __init__(self):
        self.pos_per_user = None
        self.nUsers = 0
        self.nItems = 0
        self.nClicks = 0

        self.userids = {}
        self.itemids = {}

        self.rUserids = {}
        self.rItemids={}

        self.pos_per_user={}

    # def loadData(self, filepath, userMin, itemMin):

    def loadClicks(self, filepath, userMin, itemMin):

        uCounts={}
        iCounts={}
        nRead=0
        print("  Loading clicks from %s, userMin = %d  itemMin = %d ", filepath, userMin, itemMin)

        with open(filepath, 'r') as f:
            for line in f.readlines():
                uName, iName, rating, time= line.strip().split(' ')
                nRead+=1
                if uName not in uCounts:
                    uCounts[uName]=0
                if iName not in iCounts:
                    iCounts[iName]=0

                uCounts[uName]+=1
                iCounts[iName]+=1
        print("\n  First pass: #users = %d, #items = %d, #clicks = %d\n",
        len(uCounts),len(iCounts), nRead)



        with open(filepath, 'r') as f:
            for line in f.readlines():
                uName, iName, rating, time= line.strip().split(' ')
                try:
                    tmp = int(time)
                except:
                    continue

                if uCounts[uName]< userMin:
                    continue

                if iCounts[iName]< itemMin:
                    continue

                self.nClicks+=1

                if  iName not in self.itemids:
                    self.rItemids[self.nItems]=iName
                    self.itemids[iName]=self.nItems
                    self.nItems+=1

                if uName not in self.userids:
                    self.rUserids[self.nUsers]=uName
                    self.userids[uName]=self.nUsers
                    self.nUsers+=1
                    self.pos_per_user[self.userids[uName]]=[]
                self.pos_per_user[self.userids[uName]].append((self.itemids[iName], int(time)))

            print("  Sorting clicks for each users ")

            for u in range(self.nUsers):
                sorted(self.pos_per_user[u], key=lambda d: d[1])
            print("\n \"nUsers\": %d,\"nItems\":%d, \"nClicks\":%d\n",self.nUsers,self.nItems,self.nClicks)

            self.val_per_user = []
            self.test_per_user = []
            self.train_per_user={}
            self.test_negative_per_user = {}
            mat = sp.dok_matrix((self.nUsers, self.nItems), dtype=np.float32)
            np.random.seed(2017)
            for u in range(self.nUsers):

                if len(self.pos_per_user[u])<3:
                    item_test=-1
                    item_valid=-1
                    continue

                item_test=self.pos_per_user[u][-1][0]
                self.pos_per_user[u].pop()
                item_valid=self.pos_per_user[u][-1][0]
                self.pos_per_user[u].pop()

                # item_val=self.pos_per_user[u][-1][0]
                # self.pos_per_user[u].pop()
                # self.train_per_user[u]=[]
                self.train_per_user[u]=[e[0] for e in self.pos_per_user[u]]

                self.test_per_user.append([u,item_test])
                # self.test_per_user[u]=item_test
                self.val_per_user.append([u,item_valid])

                for item in self.train_per_user[u]:
                    mat[u,item]=1.0

                self.test_negative_per_user[u]=[]
                for i in range(100):
                    neg_item_id = np.random.randint(0,self.nItems)
                    while neg_item_id in self.train_per_user[u] or neg_item_id==item_test \
                          or neg_item_id==item_valid or neg_item_id in self.test_negative_per_user[u]:
                        neg_item_id = np.random.randint(0, self.nItems)
                    self.test_negative_per_user[u].append(neg_item_id)
            self.trainMatrix = mat
            self.testRatings = self.test_per_user
            self.validRatings = self.val_per_user

            self.testNegatives=[]
            for u in range(self.nUsers):
                if u in self.train_per_user:
                    self.testNegatives.append([e for e in self.test_negative_per_user[u]])

