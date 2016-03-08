
# coding: utf-8
'''
Author:Qijie Pan

This recommendation is based on spark's MLLib ALS function and the data set is from movie lens.

'''


import math
# read the movies data and define the partition,here we split it into 2 partition
ratingsFilename = 'Data/ratings.dat'
moviesFilename = 'Data/movies.dat'

def get_ratings_tuple(entry):
    """ Parse a line in the ratings dataset
    Args:
        entry (str): a line in the ratings dataset in the form of UserID::MovieID::Rating::Timestamp
    Returns:
        tuple: (UserID, MovieID, Rating)
    """
    items = entry.split('::')
    return int(items[0]), int(items[1]), float(items[2])

def get_movie_tuple(entry):
    """ Parse a line in the movies dataset
    Args:
        entry (str): a line in the movies dataset in the form of MovieID::Title::Genres
    Returns:
        tuple: (MovieID, Title)
    """
    items = entry.split('::')
    return int(items[0]), items[1]


numPartitions = 2
rawRatings = sc.textFile(ratingsFilename).repartition(numPartitions)
rawMovies = sc.textFile(moviesFilename).repartition(numPartitions)
ratingsRDD = rawRatings.map(get_ratings_tuple).cache()
moviesRDD = rawMovies.map(get_movie_tuple).cache()



# Given two ratings RDDs, *x* and *y* of size *n*, we define RSME as follows: $ RMSE = \sqrt{\frac{\sum_{i = 1}^{n} (x_i - y_i)^2}{n}}$


trainingRDD, validationRDD, testRDD = ratingsRDD.randomSplit([6, 2, 2], seed=0L)

print 'Training: %s, validation: %s, test: %s\n' % (trainingRDD.count(),
                                                    validationRDD.count(),
                                                    testRDD.count())
print trainingRDD.take(3)
print validationRDD.take(3)
print testRDD.take(3)



def computeError(predictedRDD, actualRDD):
    """ Compute the root mean squared error between predicted and actual
    Args:
        predictedRDD: predicted ratings for each movie and each user where each entry is in the form
                      (UserID, MovieID, Rating)
        actualRDD: actual ratings where each entry is in the form (UserID, MovieID, Rating)
    Returns:
        RSME (float): computed RSME value
    """
    # Transform predictedRDD into the tuples of the form ((UserID, MovieID), Rating)
    predictedReformattedRDD = predictedRDD.map(lambda x:((x[0],x[1]),x[2]))

    # Transform actualRDD into the tuples of the form ((UserID, MovieID), Rating)
    actualReformattedRDD = actualRDD.map(lambda x:((x[0],x[1]),x[2]))

    # Compute the squared error for each matching entry (i.e., the same (User ID, Movie ID) in each
    # RDD) in the reformatted RDDs using RDD transformtions - do not use collect()
    squaredErrorsRDD = (predictedReformattedRDD
                        .join(actualReformattedRDD).map(lambda x:(x[0],math.pow(x[1][0]-x[1][1],2))))
    
    # Compute the total squared error - do not use collect()
    totalError = squaredErrorsRDD.reduce(lambda x1,x2:("None",x1[1]+x2[1]))[1]
    
    # Count the number of entries for which you computed the total squared error
    numRatings = squaredErrorsRDD.count()
    
    # Using the total squared error and the number of entries, compute the RSME
    return math.sqrt(totalError*1.0/numRatings)



#For here, we choose different rank to build the model.Find the rank 4 is the best 
from pyspark.mllib.recommendation import ALS

validationForPredictRDD = validationRDD.map(lambda x:(x[0],x[1]))

seed = 5L
iterations = 5
regularizationParameter = 0.1
ranks = [4, 8, 12]
errors = [0, 0, 0]
err = 0
tolerance = 0.03

minError = float('inf')
bestRank = -1
bestIteration = -1
for rank in ranks:
    model = ALS.train(trainingRDD, rank, seed=seed, iterations=iterations,
                      lambda_=regularizationParameter)
    predictedRatingsRDD = model.predictAll(validationForPredictRDD)
    error = computeError(predictedRatingsRDD, validationRDD)
    errors[err] = error
    err += 1
    print 'For rank %s the RMSE is %s' % (rank, error)
    if error < minError:
        minError = error
        bestRank = rank

print 'The best model was trained with rank %s' % bestRank



# this is used to test the model
myModel = ALS.train(trainingRDD,rank=8,seed = seed, iterations=iterations,lambda_=regularizationParameter)
testForPredictingRDD = testRDD.map(lambda x:(x[0],x[1]))
predictedTestRDD = myModel.predictAll(testForPredictingRDD)



# #### The user ID 0 is unassigned, so we will use it for your ratings. We set the variable `myUserID` to 0 for you. Next, create a new RDD `myRatingsRDD` with your ratings for at least 10 movie ratings. Each entry should be formatted as `(myUserID, movieID, rating)` (i.e., each entry should be formatted in the same way as `trainingRDD`).  As in the original dataset, ratings should be between 1 and 5 (inclusive). If you have not seen at least 10 of these movies, you can increase the parameter passed to `take()` in the above cell until there are 10 movies that you have seen (or you can also guess what your rating would be for movies you have not seen).


myUserID = 0

# Note that the movie IDs are the *last* number on each line. A common error was to use the number of ratings as the movie ID.
myRatedMovies = [
     (0,1193,4.5),
     (0,914,2.5),
     (0,2355,3.0),
     (0,1287,4.0),
     (0,594,4.5),
     (0,595,4.0),
     (0,2398,4.5),
     (0,1035,4.0),
     (0,2687,3.5),
     (0,3105,4.0),
     # The format of each line is (myUserID, movie ID, your rating)
     # For example, to give the movie "Star Wars: Episode IV - A New Hope (1977)" a five rating, you would add the following line:
     #   (myUserID, 260, 5),
    ]
myRatingsRDD = sc.parallelize(myRatedMovies)
print 'My movie ratings: %s' % myRatingsRDD.take(10)

#add my data to the training dataset
trainingWithMyRatingsRDD = trainingRDD.union(myRatingsRDD)


#movie recommendation model
myRatingsModel = ALS.train(trainingWithMyRatingsRDD, bestRank, seed = seed,iterations = iterations,lambda_=regularizationParameter)



# Use the Python list myRatedMovies to transform the moviesRDD into an RDD with entries that are pairs of the form (myUserID, Movie ID) and that does not contain any movies that you have rated.
myUnratedMoviesRDD = (moviesRDD
                      .filter(lambda x:x[0] not in [b for (a,b,c) in myRatedMovies]).map(lambda x:(0,x[0])))

# Use the input RDD, myUnratedMoviesRDD, with myRatingsModel.predictAll() to predict your ratings for the movies
predictedRatingsRDD = myRatingsModel.predictAll(myUnratedMoviesRDD)



def getCountsAndAverages(IDandRatingsTuple):
    """ Calculate average rating
    Args:
        IDandRatingsTuple: a single tuple of (MovieID, (Rating1, Rating2, Rating3, ...))
    Returns:
        tuple: a tuple of (MovieID, (number of ratings, averageRating))
    """
    sumRating = sum(IDandRatingsTuple[1])
    averageRating = sumRating*1.0/len(IDandRatingsTuple[1])
    return (IDandRatingsTuple[0],(len(IDandRatingsTuple[1]),averageRating))

movieIDsWithRatingsRDD = (ratingsRDD
                          .map(lambda x:(x[1],x[2])).groupByKey())

movieIDsWithAvgRatingsRDD = movieIDsWithRatingsRDD.map(getCountsAndAverages)


# Transform movieIDsWithAvgRatingsRDD from part (1b), which has the form (MovieID, (number of ratings, average rating)), into and RDD of the form (MovieID, number of ratings)
movieCountsRDD = movieIDsWithAvgRatingsRDD.map(lambda x:(x[0],x[1][0]))

# Transform predictedRatingsRDD into an RDD with entries that are pairs of the form (Movie ID, Predicted Rating)
predictedRDD = predictedRatingsRDD.map(lambda x:(x[1],x[2]))

# Use RDD transformations with predictedRDD and movieCountsRDD to yield an RDD with tuples of the form (Movie ID, (Predicted Rating, number of ratings))
predictedWithCountsRDD  = (predictedRDD
                           .join(movieCountsRDD))

# Use RDD transformations with PredictedWithCountsRDD and moviesRDD to yield an RDD with tuples of the form (Predicted Rating, Movie Name, number of ratings), for movies with more than 75 ratings
ratingsWithNamesRDD = (predictedWithCountsRDD
                       .join(moviesRDD).map(lambda x:(x[1][0][0],x[1][1],x[1][0][1])).filter(lambda x:x[2]>75))

predictedHighestRatedMovies = ratingsWithNamesRDD.takeOrdered(20, key=lambda x: -x[0])
print ('My highest rated movies as predicted (for movies with more than 75 reviews):\n%s' %
        '\n'.join(map(str, predictedHighestRatedMovies)))





