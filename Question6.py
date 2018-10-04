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

#Just proof that the samples exist, and are random
visualize(A0[0])
print("0")
visualize(A1[0])
print("1")
visualize(A2[0])
print("2")
visualize(A3[0])
print("3")
visualize(A4[0])
print("4")
visualize(A5[0])
print("5")
visualize(A6[0])
print("6")
visualize(A7[0])
print("7")
visualize(A8[0])
print("8")
visualize(A9[0])
print("9")

#SVD of the each sample space
#Gives us the subspaces for each digit
U0,s0,V0 = np.linalg.svd(A0)
U1,s1,V1 = np.linalg.svd(A1)
U2,s2,V2 = np.linalg.svd(A2)
U3,s3,V3 = np.linalg.svd(A3)
U4,s4,V4 = np.linalg.svd(A4)
U5,s5,V5 = np.linalg.svd(A5)
U6,s6,V6 = np.linalg.svd(A6)
U7,s7,V7 = np.linalg.svd(A7)
U8,s8,V8 = np.linalg.svd(A8)
U9,s9,V9 = np.linalg.svd(A9)

identity = np.identity(28)      #identity matrix for 28x28
index = random.randint(0, 9999) #random int to get unknown digit
z = testImages[index]           #unknown digit
z_label = testLabels[index]     #label of unknown digit

#residual = 













