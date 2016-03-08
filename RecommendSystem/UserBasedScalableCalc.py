"""

As we got the fraction and Numerator, we could calculate the similarity


"""
# Todo: implement with MRJOB

from mrjob.job import MRJob
from mrjob.step import MRStep

class UserBasedScalableCalc(MRJob):

    def mapper_getTwoFile(self, _, line):
        dic = {}
        if line[1:10] == "Numerator":
            keyleftPart = line.split("\t")[0]
            keyrightPart = line.split("\t")[1]
        else:
            key = line.split("\t")[0]
            value = line.split("\t")[1]
            dic.setdefault(key,value)



