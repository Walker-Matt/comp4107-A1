# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 23:09:48 2018

@author: brand
"""

import numpy as np

A = np.array([], dtype = 'float')
X = np.array([], dtype = 'float')
#Y = np.array([])
delta = 0.001

for t in range(1,1402):
    x = -0.7 + delta*(t-1)
    X = np.append(X,x)
#    y = -0.7 + delta*(i-1)
#    Y = np.append(Y,y)
    
for i in range(1401):
    row = np.array([], dtype = 'float')
    for j in range(1401):
        a_ij = np.sqrt(1-(X[i])**2-(X[j])**2)
        row = np.append(row,a_ij)
    if i == 0:
        A = row
    else:
        A = np.vstack((A,row))

U,s,V = np.linalg.svd(A)
s = np.diag(s)
#V = np.transpose(V)

sV = np.matmul(s[0:2,0:2],V[0:2,:]) 
A2 = np.matmul(U[:,0:2],sV)
print("A2 = ", A2)

norm = np.linalg.norm(A-A2, ord=2)
print()
print("||A-A2|| = ", norm)