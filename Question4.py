############################# Question 4 #############################
import numpy as np

#a,b,c,d correspond to rows 1,2,3,4 of the matrix A
a = np.array([1,2,3], dtype = 'float')
b = np.array([2,3,4], dtype = 'float')
c = np.array([4,5,6], dtype = 'float')
d = np.array([1,1,1], dtype = 'float')

A = np.array([a,b,c,d])

delta = 0.01
epsilon = np.array([0.01,0.05,0.1,0.15,0.2,0.25,0.5])

#x = np.array([1,1,1])

At = np.transpose(A)
B = np.array([1,1,1,1])
B = np.vstack(B)
AtB = np.matmul(At,B)
iters = np.array([])

def norm(x):
    Ax = np.matmul(A,x)
    AtAx = np.matmul(At,Ax)
    AtAx = np.vstack(AtAx)
    return np.linalg.norm(AtAx-AtB, ord=2)

def converge(x):
    Ax = np.matmul(A,x)
    AtAx = np.matmul(At,Ax)
    AtAx = np.vstack(AtAx)
    return AtAx-AtB
    
leastSq = list([])

for e in epsilon:
    eps = np.array([])
    x = np.array([1,1,1])
    x = np.vstack(x)
    y = norm(x)
    while (y)>delta:
        eps = np.append(eps,x)
        print("y = ",y)
        x = x-e*converge(x)
        y = norm(x)
    #print("eps = ", eps)
    iters = np.append(iters,len(eps))
    leastSq = np.append(leastSq,eps)
#    print("iters = ",iters)
#    print("leastSq = ",leastSq)

#for i in range(len(epsilon)):
#    print(len(leastSq[i]))
    
colLen = max(iters)
#print(leastSq)
    
    
