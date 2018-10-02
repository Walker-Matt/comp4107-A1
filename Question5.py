############################# Question 5 #############################
import numpy as np
import scipy.linalg as linalg

#a,b,c,d correspond to rows 1,2,3,4 of the matrix A
a = np.array([3,2,-1,4])
b = np.array([1,0,2,3])
c = np.array([-2,-2,3,-1])

A = np.array([a,b,c])     #creates Matrix
print("Original Matrix:")
print(A)

null = linalg.null_space(A)     #calculates null space vectors

# separating vectors
null1 = np.array([null[0,0],null[1,0],null[2,0],null[3,0]])
null1 = np.vstack(null1)
null2 = np.array([null[0,1],null[1,1],null[2,1],null[3,1]])
null2 = np.vstack(null2)

print()
print("Two linearly independent vectors belonging to the null space of A:")
print(null1)
print()
print(null2)
print()

# calculates rank of columns/rows
col_rank = np.linalg.matrix_rank(A)
row_rank = np.linalg.matrix_rank(np.transpose(A))

print("Column rank = ", col_rank)
print("""Since the number of columns is greater than the column rank of A, it
is not linearly independent in R^3.""")
print()
print("Row rank = ", row_rank)
print("""Since the number of rows is greater than the row rank of A, it
is not linearly independent in R^4.""")
print()

inverse = linalg.pinv(A)        # calculates pseduo inverse of matrix A
print("Pseduo-inverse of Matrix A:")
print(inverse)