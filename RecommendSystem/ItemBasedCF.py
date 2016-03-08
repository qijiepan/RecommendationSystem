'''

This is the item based cf is useful when the number of items is smaller than the number of users.
And it's suitable for the area where user personal intersts is important 
What's more, it's friendly to new item rather than new users
Finally, it's convincing and can explain why recommeny item to user
'''

import math

class ItemBased(object):
    def __init__(self,train_file,test_file):
        self.train_file = train_file
        self.test_file = test_file
        self.readData()
        self.__itemSimilarity() #calculate similarity

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

    def __itemSimilarity(self):
        C = {}
        N = {}
        for user,items in self.train.items():
            for i in items:
                N.setdefault(i,0)
                N[i] +=1
                C.setdefault(i,{})
                for j in items:
                    if i == j:
                        continue
                    C[i].setdefault(j,0)
                    C[i][j]+=1

        self.W= {}
        for i,related_items in C.items():
            self.W.setdefault(i,{})
            for j,cjj in related_items.items():
                self.W[i][j] = cjj/(math.sqrt(N[i]*N[j]))

    def Recommend(self,user,K=3,N=10):
        rank = {}
        action_item =self.train[user]
        for item,score in action_item.items():
            for j,wj in sorted(self.W[item].items(),key=lambda x:x[1],reverse=True)[0:K]:
                if j in action_item:
                    continue
                rank.setdefault(j,0)
                rank[j] += score * wj
        return dict(sorted(rank.items(),key=lambda x:x[1],reverse=True)[0:N])

