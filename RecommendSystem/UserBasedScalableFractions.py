'''

For this map reduce job, we get the the number of movies both user u and user v seen
But be careful, we divide by the popularity of this movie as a penalty.


'''
# Todo: implement with MRJOB

from mrjob.job import MRJob
from mrjob.step import MRStep

import math


class UserBasedScalableFractions(MRJob):
    def steps(self):
        return [
            MRStep(mapper=self.mapper_get_user_item_score,
                   reducer=self.reducer_User),
            MRStep(mapper=self.mapper_getUserSimilarity,
                   reducer=self.reducer_getCUV)
        ]

    def mapper_get_user_item_score(self, _, line):
        if not line.strip():
            pass
        else:
            user = line.strip().split("\t")[0]
            item = line.strip().split("\t")[1]
            score = line.strip().split("\t")[2]
            yield item, (user, score)

    def reducer_User(self, item, user_score):
        item_userlist = {}
        for user, score in user_score:
            item_userlist.setdefault(item, [])
            item_userlist[item].append(user + "::" + score)
        for i, ulist in item_userlist.items():
            yield i, ulist

    # Now we key such key value pairs:(item:[user1::score1,[user2::score2]])
    def mapper_getUserSimilarity(self, _, user_score):
        user_list = []
        for u_s in user_score:
            user, score = u_s.split("::")
            user_list.append(user)
        for u in user_list:
            for v in user_list:
                if u != v:
                    yield (u + "::" + v), (1.0 / math.log(1 + (len(user_list)))) #this is the penalty

    def reducer_getCUV(self, UV, CUV):
        yield UV, sum(CUV)



if __name__ == '__main__':
    UserBasedScalableFractions.run()
