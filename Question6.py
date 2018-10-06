############################# Question 6 #############################
import numpy as np
import gzip
import struct
import array
import random

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

#Visualizer for images matrices, just to see
def visualize(matrix):
    for i in range(27):
        line = ""
        for j in range(27):
            if matrix[i][j] == 0:
                line = line + " -"
            else:
                line = line + " #"
        print(line)
        
#Populates sample arrays with 50 images of desired digit
def randomDigitSet(digit):
    #digitSet = np.zeros((50,28,28))
    digitSet = np.linspace(0,783,784)
    count = 0
    while(count < 50):
        index = random.randint(0, 59999)
        if(trainingLabels[index] == digit):
            image = np.transpose(trainingImages[index])
            tempArray = np.array([])
            for i in range(len(image)):
                tempArray = np.append(tempArray, image[i])
            digitSet = np.vstack((digitSet,tempArray))
            #digitSet[count] = trainingImages[index]
            count = count + 1
    digitSet = digitSet[1:len(digitSet)]
    return np.transpose(digitSet)
        

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

##Just proof that the samples exist, and are random
#visualize(A0)#[0])
#print("0")
#visualize(A1[0])
#print("1")
#visualize(A2[0])
#print("2")
#visualize(A3[0])
#print("3")
#visualize(A4[0])
#print("4")
#visualize(A5[0])
#print("5")
#visualize(A6[0])
#print("6")
#visualize(A7[0])
#print("7")
#visualize(A8[0])
#print("8")
#visualize(A9[0])
#print("9")

def getU(sampleSpace, size):
    U, s, V = np.linalg.svd(sampleSpace[0:size], full_matrices = False)
    return U

def residual(ident, U, z):
    UUt = np.matmul(U,np.transpose(U))
    print("identity = ", ident.shape)
    print("UUt = ", UUt.shape)
    diff = ident - UUt
    diffZ = np.matmul(diff,z)
    return np.linalg.norm(diffZ, ord=2)

#zVals = np.append([])
#for i in range(1000):
#    index = random.randint(0, 9999) #random int to get unknown digit
#    z = testImages[index]           #unknown digit
#    z_label = testLabels[index]     #label of unknown digit
    
percent = np.array([])

for k in range(1,52):
    U0 = getU(A0, k)
    U1 = getU(A1, k)
    U2 = getU(A2, k)
    U3 = getU(A3, k)
    U4 = getU(A4, k)
    U5 = getU(A5, k)
    U6 = getU(A6, k)
    U7 = getU(A7, k)
    U8 = getU(A8, k)
    U9 = getU(A9, k)
    
    correct = 0
    identity = np.identity(k)      #identity matrix for k x k
    U = np.array([U0,U1,U2,U3,U4,U5,U6,U7,U8,U9])
    
    for j in range(len(testImages)):
        image = testImages[j]
        z_label = testLabels [j]
        residuals = np.array([])
        
        z = np.transpose(image)
        tempArray = np.array([])
        for i in range(len(z)):
            tempArray = np.append(tempArray, image[i])
        z = tempArray
        
        for u in U:
            res = residual(identity,u,z)
            residuals = np.append(residuals,res)
        minPos = residuals.argmin()
        if minPos == z_label:
            correct = correct + 1
    per = correct/len(testImages)
    percent = np.append(percent,per)