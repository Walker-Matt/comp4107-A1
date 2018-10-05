############################# Question 1 #############################
import numpy as np
import scipy.spatial.distance as dist

#each row of the given matrix
Alice = np.array([5,3,4,4,None], dtype = 'float')
Alicia = np.array([3,1,2,3,None], dtype = 'float')
Bob = np.array([4,3,4,3,5], dtype = 'float')
Mary = np.array([3,2,1,5,4], dtype = 'float')
Sue = np.array([1,6,5,2,1], dtype = 'float')

names = np.array(["Alice","Alicia","Bob","Mary","Sue"])
Matrix = np.array([Alice,Alicia,Bob,Mary,Sue]) #2D array
subMatrix = Matrix[1:,:-1] #removes first row and column of Matrix

U,s,V = np.linalg.svd(subMatrix, full_matrices = False)
s_inv = np.linalg.inv(np.diag(s))
Us_inv = np.matmul(U[:,0:2],s_inv[0:2,0:2])

Alice2D = np.matmul(Alice[:-1],Us_inv)
Alicia2D = np.matmul(Alicia[:-1],Us_inv)
Bob2D = np.matmul(Bob[:-1],Us_inv)
Mary2D = np.matmul(Mary[:-1],Us_inv)
Sue2D = np.matmul(Sue[:-1],Us_inv)
matrix2D = np.array([Alice2D,Alicia2D,Bob2D,Mary2D,Sue2D])

print("2D Arrays:")
for i in range(len(names)):
    print(names[i],matrix2D[i])

AA_2D = dist.euclidean(Alice2D,Alicia2D)
AB_2D = dist.euclidean(Alice2D,Bob2D)
AM_2D = dist.euclidean(Alice2D,Mary2D)
AS_2D = dist.euclidean(Alice2D,Sue2D)

AA_4D = dist.euclidean(Alice[:-1],Alicia[:-1])
AB_4D = dist.euclidean(Alice[:-1],Bob[:-1])
AM_4D = dist.euclidean(Alice[:-1],Mary[:-1])
AS_4D = dist.euclidean(Alice[:-1],Sue[:-1])

eucs_2D = np.array([AA_2D,AB_2D,AM_2D,AS_2D])
eucs_4D = np.array([AA_4D,AB_4D,AM_4D,AS_4D])

print()
print("Euclidean Distance from Alice (2D):")
for i in range(1,len(names)):
    print(names[i],eucs_2D[i-1])

print()
print("Euclidean Distance from Alice (4D):")
for i in range(1,len(names)):
    print(names[i],eucs_4D[i-1])

closest2D = min(AA_2D,AB_2D,AM_2D,AS_2D)
closest4D = min(AA_4D,AB_4D,AM_4D,AS_4D)

print()
for i in range(len(eucs_2D)):
    if closest2D == eucs_2D[i]:
        print("Closest in 2D = ", names[i+1])
        break

for i in range(len(eucs_4D)):
    if closest4D == eucs_4D[i]:
        print("Closest in 4D = ", names[i+1])
        break