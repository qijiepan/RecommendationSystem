
'''

For this map reduce job, we get the number of movies one user has watched


'''

# Todo: implement with MRJOB

from mrjob.job import MRJob
from mrjob.step import MRStep


class UserBasedScalableNumerator(MRJob):

    def steps(self):
        return [
            MRStep(mapper=self.mapper_getUser,
                   reducer=self.reducer_countMoiveForUser)
        ]

    def mapper_getUser(self,_,line):
        if not line.strip():
            pass
        else:
            user = line.strip().split("\t")[0]
            yield user, 1

    def reducer_countMoiveForUser(self, user, count):
        yield "Numerator",(user,sum(count))







if __name__ == '__main__':
    UserBasedScalableNumerator.run()

