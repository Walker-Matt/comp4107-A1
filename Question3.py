############################# Question 3 #############################
import numpy as np

A = np.array([], dtype = 'float')
X = np.array([], dtype = 'float') #since X and Y are the same, can just use X
delta = 0.001

for t in range(1,1402):
    x = -0.7 + delta*(t-1)
    X = np.append(X,x)
    
for i in range(1401):
    row = np.array([], dtype = 'float')
    for j in range(1401):
        a_ij = np.sqrt(1-(X[i])**2-(X[j])**2) #given function for a_ij elements
        row = np.append(row,a_ij) #builds i-th row
    if i == 0:
        A = row
    else:
        A = np.vstack((A,row)) #builds the matrix by stacking row-by-row

U,s,V = np.linalg.svd(A)
s = np.diag(s)

sV = np.matmul(s[0:2,0:2],V[0:2,:]) #first two rows/columns of s and only first two rows of V but all columns
A2 = np.matmul(U[:,0:2],sV) #first two columns of U but all rows
print("A2 = ", A2)

norm = np.linalg.norm(A-A2, ord=2)
print()
print("||A-A2|| = ", norm)