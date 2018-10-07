############################# Question 7 #############################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

k = 8
threshold = np.array([300,400,500,600,700,800])
X = np.array([0.2, 0.5, 0.8])

data = pd.read_csv("ml-latest-small//ratings.csv", header = 0)
userID = data[['userId']].values
movieID = data[['movieId']].values
rating = data[['rating']].values

movies = list(np.unique(movieID))
numMovies = len(movies)
numUsers = max(userID)[0]

movieDict = dict()
for i in range(numMovies):
    movieDict[movies[i]] = i

num = len(data)
index = np.linspace(0,num-1,num, dtype = "int")
np.random.shuffle(index)

yvals = np.zeros(len(threshold))

for x in X:
    MAEvals = np.array([])
    for thresh in threshold:
        numTrain = int(x*num)
        #numTest = num - numTrain
        train = index[0:numTrain]
        test = index[numTrain:]
        
        trainSet = np.zeros((numUsers, numMovies))
        testSet = np.zeros((numUsers, numMovies))
        
        for j in train:
            user = userID[j][0]-1
            moviePos = movieDict[movieID[j][0]]
            trainSet[user][moviePos] = rating[j][0]
            
        for t in test:
            user = userID[t][0]-1
            moviePos = movieDict[movieID[t][0]]
            testSet[user][moviePos] = rating[t][0]
        
        A = trainSet[0:thresh]
        U,s,V = np.linalg.svd(A, full_matrices = False)
        U_k = U[:,0:k]
        s_k = np.diag(s[0:k])
        V_k = V[0:k,:]
        
        s_sqrt = np.sqrt(s_k)
        
        Vs = np.matmul(np.transpose(V_k), np.linalg.inv(s_k))
        
        for b in range(thresh, numUsers):
            t = trainSet[b]
            newFold = np.matmul(t,Vs)
            U_k = np.vstack((U_k, newFold))
        
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
    yvals = np.vstack((yvals,MAEvals))

yvals = yvals[1:]

plt.figure()
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
plt.plot(threshold,yvals[0], label = "x = 0.2")
plt.plot(threshold,yvals[1], label = "x = 0.5")
plt.plot(threshold,yvals[2], label = "x = 0.8")
plt.title("Model-based SVD Prediction using Folding-in")
plt.xlabel("Folding-in Model Size")
plt.ylabel("MAE")
plt.legend()
plt.grid()
plt.show()
