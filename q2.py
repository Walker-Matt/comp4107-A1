############################# Question 2 #############################
import numpy as np

#a,b,c,d correspond to rows 1,2,3,4 of the matrix A
a = np.array([1,2,3], dtype = 'float')
b = np.array([2,3,4], dtype = 'float')
c = np.array([4,5,6], dtype = 'float')
d = np.array([1,1,1], dtype = 'float')

A = np.array([a,b,c,d])

U,s,V = np.linalg.svd(A, full_matrices = False)

print("U = ")
print(U)
print("s = ")
print(np.diag(s))
print("V = ")
print(V)