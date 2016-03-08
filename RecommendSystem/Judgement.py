'''
This is the function to make sure the recommend system is good/bad

Based on Precision, Recall, Coverage and Popularity

Author:Qijie Pan

'''

import math
from multiprocessing import pool
from UserBasedCF import UserBased
from ItemBasedCF import ItemBased



class Judgement(object):
    def __init__(self,method = 'UserBased',train = None,test = None): #default we choose user-based recommend system
        self.__precision = 0
        self.__recall = 0
        self.__coverage = 0
        self.__popularity = 0
        if method == 'UserBased':
            self.__algorithm = UserBased(train,test)
        if method == 'ItemBased':
            self.__algorithm = ItemBased(train,test)
        self.__calculateCoverage()
        self.__calculatePopularity()
        self.__calculatePrecisionAndRecall()

    def __calculatePrecisionAndRecall(self,K =3,N =10):

        hit = 0
        for user in self.__algorithm.train.keys():
            tu = self.__algorithm.test.get(user,{})
            rank = self.__algorithm.Recommend(user,K = K,N = N)
            for i,_ in rank.items():
                if i in tu:
                    hit +=1
            self.__recall += len(tu)
            self.__precision += N
        self.__recall = hit/(self.__recall *1.0)
        self.__precision = hit/(self.__precision *1.0)

    def __calculateCoverage(self,K=3,N=10):
        recommend_items = set()
        all_items = set()
        for user,items in self.__algorithm.train.items():
            for i in items.keys():
                all_items.add(i)
            rank  = self.__algorithm.Recommend(user,K)
            for i,_ in rank.items():
                recommend_items.add(i)
        self.__coverage = len(recommend_items)/(len(all_items) * 1.0)

    def __calculatePopularity(self,K=3,N=10):
        item_popularity = {}
        for user,items in self.__algorithm.train.items():
            for i in items.keys():
                item_popularity.setdefault(i,0)
                item_popularity[i] +=1

        ret = 0 # result of popularity
        n = 0 # the total amount of recommendation
        for user in self.__algorithm.train.keys():
            rank = self.__algorithm.Recommend(user,K=K,N=N)
            for item,_ in rank.items():
                ret += math.log(1+item_popularity[item]) #use log here because we define 10 as popular baseline
                n +=1
        ret /= n*1.0
        self.__popularity = ret

    def getPrecision(self):
        return self.__precision

    def getRecall(self):
        return self.__recall

    def getPopularity(self):
        return self.__popularity

    def getCoverage(self):
        return self.__coverage

if __name__ == '__main__':
    test = Judgement(method='UserBased',train = 'u1.base',test='u1.test')
    print test.getPrecision(),test.getCoverage(),test.getPopularity(),test.getRecall()








