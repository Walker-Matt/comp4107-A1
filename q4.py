############################# Question 4 #############################
import numpy as np
import math
import pandas as pd

#a,b,c,d correspond to rows 1,2,3,4 of the matrix A
a = np.array([1,2,3], dtype = 'float')
b = np.array([2,3,4], dtype = 'float')
c = np.array([4,5,6], dtype = 'float')
d = np.array([1,1,1], dtype = 'float')

A = np.array([a,b,c,d])     #creates Matrix

delta = 0.01
epsilon = np.array([0.01,0.05,0.1,0.15,0.2,0.25,0.5])
xvals = np.array([])
iters = np.array([], dtype = int)

At = np.transpose(A)
B = np.array([1,1,1,1])
B = np.vstack(B)            #transposes an array from row to column
AtB = np.matmul(At,B)       #matrix multiplication
AtA = np.matmul(At,A)

def norm(x):
    AtAx = np.matmul(AtA,x)
    return np.linalg.norm(AtAx-AtB, ord=2)  #calculates order 2 norm

def diff(x):
    AtAx = np.matmul(AtA,x)
    return AtAx-AtB

for eps in epsilon:
    numIters = 0
    x = np.array([1,1,1])   #starting x values
    x = np.vstack(x)
    y = norm(x)
    while y>delta:          #while the calculated norm is greater than delta
        x = x-eps*diff(x)   #adjusts x value
        y = norm(x)         #updates the norm with new x value
        numIters+=1
    iters = np.append(iters, numIters)
    if math.isnan(y):       #check if y diverged
        x = np.array(["----", "Diverged...", "----"])
    else:
        x = np.hstack(x)    #transposes an array from column to row
        x = np.around(x,4)  #rounds x values to 4 decimal places
    if eps == 0.01:
        xvals = np.array([x])
    else:
        xvals = np.vstack((xvals,x))
        
data = {'Epsilon': epsilon, 
        'Min X Values      ': xvals.tolist(), 
        '# Iterations': iters}
table = pd.DataFrame(data=data)
print(table)