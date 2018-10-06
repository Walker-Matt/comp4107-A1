############################# Question 7 #############################
import numpy as np
import pandas as pd

data = pd.read_csv("ml-latest-small//ratings.csv", header = 0)
userID = data[['userId']].values
movieID = data[['movieId']].values
rating = data[['rating']].values

movies = list(np.unique(movieID))
numMovies = len(movies)

movieDict = dict()
for i in range(numMovies):
    movieDict[movies[i]] = i-1

num = len(data)
index = np.linspace(0,num-1,num, dtype = "int")
np.random.shuffle(index)

x = 0.8 #np.array([0.2, 0.5, 0.8])
numTrain = int(x*len(data))
numTest = len(data) - numTrain
train = index[0:numTrain]
test = index[numTrain:]

train_set = np.zeros((max(userID)[0], numMovies))
for j in train:
    user = userID[j][0]-1
    moviePos = movieDict[movieID[j][0]]
    train_set[user][moviePos] = rating[j][0]

#numTestMovies = len(np.unique(movieID[test]))
test_set = np.zeros((max(userID)[0], numMovies))
for k in test:
    user = userID[k][0]-1
    moviePos = movieDict[movieID[k][0]]
    test_set[user][moviePos] = rating[k][0]   