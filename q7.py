############################# Question 7 #############################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import time

k = 14
threshold = np.array([200,250,300,350,400,450,500,550,600])#650,700,750,800])
X = np.array([0.2, 0.5, 0.8])

data = pd.read_csv("ml-latest-small//ratings.csv", header = 0)
userID = data[['userId']].values
movieID = data[['movieId']].values
rating = data[['rating']].values

movies = list(np.unique(movieID)) # list of unique movieId's (All movies)
numMovies = len(movies)
numUsers = max(userID)[0]

# maps a movieId to a column number
movieDict = dict()
for i in range(numMovies):
    movieDict[movies[i]] = i

#creates a randomly shuffled order of indices to separate train/test sets
num = len(data)
index = np.linspace(0,num-1,num, dtype = "int")
np.random.shuffle(index)

yvals = np.zeros(len(threshold))
timeVals = np.zeros(len(threshold))
for x in X:
    MAEvals = np.array([])
    predTimes = np.array([])
    numTrain = int(x*num)       # number of ratings to use for training
    train = index[0:numTrain]   # takes first "numTrain" of shuffled indices
    test = index[numTrain:]     # remaining are used for testing
    trainSet = np.zeros((numUsers, numMovies))
    testSet = np.zeros((numUsers, numMovies))
    
    for j in train:
        user = userID[j][0]-1                   # gets userID at index j as row number
        moviePos = movieDict[movieID[j][0]]     # gets movieID at index j and maps to column number
        trainSet[user][moviePos] = rating[j][0] # puts rating at index j in position of row,column
        
    for t in test:
        user = userID[t][0]-1
        moviePos = movieDict[movieID[t][0]]
        testSet[user][moviePos] = rating[t][0]
    

    for thresh in threshold: 
        start = time.time()
        A = trainSet[0:thresh]      # uses first "threshold" number of rows for original matrix
        
        # changes zero values to the average rating of that given user    
        for row in range(thresh):
            rowAvg = np.mean(trainSet[row,:])
            for col in range(numMovies):
                if trainSet[row,col] == 0:
                    trainSet[row,col] = rowAvg;
        
        U,s,V = np.linalg.svd(A, full_matrices = False)
        U_k = U[:,0:k]
        s_k = np.diag(s[0:k])
        V_k = V[0:k,:]
        Vs = np.matmul(np.transpose(V_k), np.linalg.inv(s_k))
        
        # folding-in process
        for b in range(thresh, numUsers):
            t = trainSet[b]
            newFold = np.matmul(t,Vs)
            U_k = np.vstack((U_k, newFold))
        
        s_sqrt = np.sqrt(s_k)
        Us = np.matmul(U_k, np.transpose(s_sqrt))    
        sV = np.matmul(s_sqrt, V_k)
        
        prediction = np.zeros((numUsers, numMovies))
        for q in test:
            row = userID[q][0]-1
            col = movieDict[movieID[q][0]]
            rowAvg = np.mean(trainSet[row,:])
            i = Us[row,:]
            j = sV[:,col]
            prediction[row][col] = rowAvg + np.dot(i,j)    
        MAE = sum(sum(np.absolute(prediction-testSet)))/len(test)
        MAEvals = np.append(MAEvals, MAE)
        end = time.time()
        elapsedTime = end-start
        pTime = len(test)/elapsedTime
        predTimes = np.append(predTimes,pTime)
    yvals = np.vstack((yvals,MAEvals))
    timeVals = np.vstack((timeVals,predTimes))

yvals = yvals[1:]
timeVals = timeVals[1:]

plt.figure()
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
plt.plot(threshold,yvals[0], label = "x = 0.2", marker = "o", color = "C0")
plt.plot(threshold,yvals[1], label = "x = 0.5", marker = "o", color = "C1")
plt.plot(threshold,yvals[2], label = "x = 0.8", marker = "o", color = "C2")
plt.title("Model-based SVD Prediction using Folding-in")
plt.xlabel("Folding-in Model Size")
plt.ylabel("MAE")
plt.legend(loc = "upper right")
plt.grid()
plt.show()

plt.figure()
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
plt.plot(threshold,timeVals[0], label = "x = 0.2", marker = "o", color = "C0")
plt.plot(threshold,timeVals[1], label = "x = 0.5", marker = "o", color = "C1")
plt.plot(threshold,timeVals[2], label = "x = 0.8", marker = "o", color = "C2")
plt.title("Throughput vs. Folding-in Basis Size")
plt.xlabel("Folding-in Model Size")
plt.ylabel("Throughput (Predictions/sec)")
plt.legend()
plt.grid()
plt.show()