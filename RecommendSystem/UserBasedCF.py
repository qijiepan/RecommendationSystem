'''
For the UserBased Algorithm, it's useful when the amount of user is not large.
And it's time efficiency.

What's more, this user-based

# I need to consider how to implement this algorith in map-reduce platform.
# Actually, if we use spark, we could use Alternating Least Square to calculate the score for the movie the user have not watched.
# For the spark reference:http://spark.apache.org/docs/latest/mllib-collaborative-filtering.html

# For here, we used Map-reduce to calculate this kind of job


'''

import collections
import math


class UserBased(object):
    def __init__(self,train_file,test_file):
        self.train_file = train_file
        self.test_file = test_file
        self.readData()
        self.__userSimilarity() #calculate similarity

    def readData(self):
        self.train = {}     #train dataset
        with open(self.train_file) as f:
            for line in f:
                user,item,score,_ = line.strip().split("\t")
                self.train.setdefault(user,{})
                self.train[user][item] = int(score)

        self.test = {}     #test dataset
        with open(self.test_file) as f:
            for line in f:
                user,item,score,_ = line.strip().split("\t")
                self.test.setdefault(user,{})
                self.test[user][item] = int(score)

    def __userSimilarity(self):
        #build the item-user reverse order table
        item_users = collections.defaultdict(set)
        for user,items in self.train.items():
            for i in items:
                item_users[i].add(user)

        #calculate the matrix between different users
        C = {}
        N = {}
        for i,users in item_users.items():
            for u in users:
                N.setdefault(u,0)
                N[u] +=1
                C.setdefault(u,{})
                for v in users:
                    if u == v:
                        continue
                    C[u].setdefault(v,0)
                    C[u][v] +=1/math.log(1+len(users)) #give a penalty to the popular item.

        self.W = dict() # similarity matrix
        for u,related_users in C.items():
            self.W.setdefault(u,{})
            for v,cuv in related_users.items():
                self.W[u][v] = cuv/ math.sqrt(N[u]*N[v])  # here we use cosine similarity, we could also use Jaccard similarity Maybe we could use pearson similarity

    def Recommend(self,user,K=3,N=10):
        rank = {}
        action_item = self.train[user].keys()  # the action made by the user
        for v,wuv in sorted(self.W[user].items(),key=lambda e:e[1],reverse=True)[0:K]:
            for i,rvi in self.train[v].items():
                if i in action_item:
                    continue
                rank.setdefault(i,0)
                rank[i] += wuv *rvi
        return dict(sorted(rank.items(),key=lambda e:e[1],reverse=True)[0:N])








