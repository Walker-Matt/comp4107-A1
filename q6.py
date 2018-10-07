############################# Question 6 #############################
import numpy as np
import gzip
import struct
import array
import random
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

# initial number of z images
# can lower to shorten run time
numZ = 50

#File names and location within the assignment folder
trainingImagesFile = 'mnist/train-images-idx3-ubyte.gz'
trainingLabelsFile = 'mnist/train-labels-idx1-ubyte.gz'
testImagesFile = 'mnist/t10k-images-idx3-ubyte.gz'
testLabelsFile = 'mnist/t10k-labels-idx1-ubyte.gz'

def parseFile(fileName):
    file = gzip.open(fileName, 'rb')
    DATA_TYPES = {0x08:'B',0x09:'b',0x0b:'h',0x0c:'i',0x0d:'f',0x0e:'d'}
    header = file.read(4)
    zeros, data_type, num_dimensions = struct.unpack('>HBB', header)
    data_type = DATA_TYPES[data_type]
    dimension_sizes = struct.unpack('>' + 'I' * num_dimensions,
                                    file.read(4 * num_dimensions))
    data = array.array(data_type, file.read())
    return np.array(data).reshape(dimension_sizes)
    
#Parsed data from the MNIST dataset files
#Arrays of 28x28 matrices and their labels
trainingImages = parseFile(trainingImagesFile)
trainingLabels = parseFile(trainingLabelsFile)
testImages = parseFile(testImagesFile)
testLabels = parseFile(testLabelsFile)
        
#Populates sample arrays with 50 images of desired digit
def randomDigitSet(digit):
    digitSet = np.zeros(784)   #produced just so np.vstack() works (removed after)
    count = 0
    while(count < 1000):
        index = random.randint(0, 59999)
        if(trainingLabels[index] == digit):
            #converts image into single column
            image = np.transpose(trainingImages[index])     #switches columns to rows
            tempArray = np.array([])
            for i in range(len(image)):
                tempArray = np.append(tempArray, image[i])  #adds rows of image together in one array
            digitSet = np.vstack((digitSet,tempArray))      #stacks rows of images on top of eachother
            count = count + 1
    digitSet = digitSet[1:len(digitSet)]    #removes first generic row of zeros
    return np.transpose(digitSet)           #switches rows back to columns

#Random sample arrays for each digit
#Initialized at proper sizes with zeros
A0 = randomDigitSet(0)
A1 = randomDigitSet(1)
A2 = randomDigitSet(2)
A3 = randomDigitSet(3)
A4 = randomDigitSet(4)
A5 = randomDigitSet(5)
A6 = randomDigitSet(6)
A7 = randomDigitSet(7)
A8 = randomDigitSet(8)
A9 = randomDigitSet(9)

identity = np.identity(784)     # identity matrix 784x784
basis = 50                      # max number of basis images
percentages = np.array([])
zIndex = np.array([], dtype = "int")

for i in range(numZ):
    index = random.randint(0, 9999)     #random int to get unknown digit
    zIndex = np.append(zIndex,index)    #stores random indices

def getU(sampleSpace):
    U, s, V = np.linalg.svd(sampleSpace, full_matrices = False)
    return U

def residual(U, z):
    UUt = np.matmul(U,np.transpose(U))
    diff = identity - UUt
    diffZ = np.matmul(diff,z)
    return np.linalg.norm(diffZ, ord=2)

U0 = getU(A0)
U1 = getU(A1)
U2 = getU(A2)
U3 = getU(A3)
U4 = getU(A4)
U5 = getU(A5)
U6 = getU(A6)
U7 = getU(A7)
U8 = getU(A8)
U9 = getU(A9)

for k in range(1,52):
    # selects k columns from the U_j matrices
    U0_k = U0[:,0:k]
    U1_k = U1[:,0:k]
    U2_k = U2[:,0:k]
    U3_k = U3[:,0:k]
    U4_k = U4[:,0:k]
    U5_k = U5[:,0:k]
    U6_k = U6[:,0:k]
    U7_k = U7[:,0:k]
    U8_k = U8[:,0:k]
    U9_k = U9[:,0:k]
    U = np.array([U0_k,U1_k,U2_k,U3_k,U4_k,U5_k,U6_k,U7_k,U8_k,U9_k])
    
    correct = 0
    for j in range(numZ):
        index = zIndex[j]
        image = testImages[index]
        z_label = testLabels[index]
        residuals = np.array([])
        
        #converts image z into single column
        imageT = np.transpose(image)
        tempArray = np.array([])
        for i in range(len(imageT)):
            tempArray = np.append(tempArray, imageT[i])
        z = np.transpose(tempArray)
        
        #computes res values of z image against each U_j
        for u in U:
            res = residual(u,z)
            residuals = np.append(residuals,res)
        minPos = residuals.argmin()         # min res is classification of digit
        if minPos == z_label:
            correct = correct + 1       # tests correctness against known label of z
    percent = 100*correct/numZ
    percentages = np.append(percentages,percent)

x = np.linspace(1,basis+1,basis+1,dtype="int")      # x-axis for graph (# of basic images)
for i in range(percentages.size):
    print(x[i], "Basis Images", " = ", percentages[i], "%")

plt.figure()
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
plt.plot(x,percentages, marker = "o", color = "C0")
plt.title("Classification Percentage vs. Number of Basis Images")
plt.xlabel("Number of Basis Images")
plt.ylabel("Classification Percentage")
plt.grid()
plt.show()