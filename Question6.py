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

#Random sample array and labels
#Initialized at proper sizes with zeros
A = np.zeros((50,28,28))
A_labels = np.zeros((50), np.int)

#Randomly populate sample array and label array
for i in range(50):
    index = random.randint(0, 59999)
    A[i] = trainingImages[index]
    A_labels[i] = trainingLabels[index]
    
#Just proof that the sample exists, and is random
visualize(A[0])
print(A_labels[0])

#SVD of the sample space
#Gives us the subspaces of a digit
U,s,V = np.linalg.svd(A)

#residual = 













