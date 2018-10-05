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
    digitSet = np.zeros((50,28,28))
    count = 0
    while(count < 50):
        index = random.randint(0, 59999)
        if(trainingLabels[index] == digit):
            digitSet[count] = trainingImages[index]
            count = count + 1
    return digitSet
        

#Random sample arrays for each digit
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

subspaces = ([A0, A1, A2, A3, A4, A5, A6, A7, A8, A9])

#Takes subspace and desired size
#Returns left-singular vectors
def getU(sampleSpace, size):
    U, s, V = np.linalg.svd(sampleSpace[0:size])
    return U

#Computes the residual between
#Left-singular vectors and unknown digit
def residual(U, z):
    ident = np.identity(28)
    UUt = np.matmul(U,np.transpose(U))
    diff = ident - UUt
    diffZ = np.matmul(diff,z)
    return np.linalg.norm(diffZ, ord=2)

#Computes average residual for one subspace
#Using entire set of test images
def averageResidual(subspace, k, z):
    residuals = []
    U = getU(subspace, k)
    for i in range(0, k):
        res = residual(U[i], z)
        residuals.append(res)
    average = np.sum(residuals) / len(residuals)
    return average

#Gathers lowest residual from set of subspaces
#Returns label of subspace with lowest residual
def lowestResidual(k, z):
    lowestRes = 1
    label = 0
    for i in range(len(subspaces)):
        if(averageResidual(subspaces[i], k, z) < lowestRes):
            lowestRes = averageResidual(subspaces[i], k, z)
            label = i
    return label

#Returns true or false if unknown image is correctly classified
def isCorrect(k, z, label):
    if(lowestResidual(k, z) == label):
        return True
    else:
        return False
    
#Returns the percentage of correct classifications
def percentage(k):
    correct = 0
    for i in range(100):
        if(isCorrect(k, testImages[i], testLabels[i])):
            correct = correct + 1
    return correct / 100

#Testing for output of results
print("Test size of 100")
for i in range(1,16):
    print(str(i) + " base images: " + repr(percentage(i)*100) + "%")
    










