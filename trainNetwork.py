'''
Sophia Davis
5/31/2014
trainNetwork.py

Implements backpropogation
'''

import sys
import csv
import math
import random
from collections import defaultdict
import pickle

def main():
    if len(sys.argv) < 2:
        sys.stderr.write('Usage: python ' + sys.argv[0] + ' trainingDataFile.csv\n')
        sys.exit(1)
    else:
        print "Reading in data...\n"
        with open(sys.argv[1], 'rb') as f:
            imagereader = csv.reader(f)
            data = []
            for row in imagereader:
                data.append(row)
        f.close()
        
        ######################## Initialization
        ####### Set network properties
        alpha = .4
        mu = 1000/(1000 + 1) # Control parameter for momentum
        testSize = 1 # Size of test set (from group of examples with each classification)
        numHiddenLayers = 1
        numNodesPerLayer = 5
        
        ####### Parse data    
        ### Collect all possible attributes and classifications
        attributes = data[0][2:] # First index is the classification, second is file name
        data = data[1:]
        sortedData = parseData(data) # Dictionary of data items sorted by classif
        classifs = sortedData.keys()

        print "Separating training and test sets..."
        testSet, trainingSet = separateTestData(sortedData, testSize)
        
        print "Training neural network on training data..."
        
        # Initialize input, hidden, and output layers    
        Network = initLayers(attributes, classifs, numHiddenLayers, numNodesPerLayer)
                
        # Forward and Backward propagate, given each data item
        random.shuffle(trainingSet) # so that all the same classifs aren't next to each other?
        for j in range(1):#00):
            print
            print "~~~~~~~~~~~~~~~~~Iteration " + str(j)
            trainingSumMSE = 0.0
            for i, ex in enumerate(trainingSet[:1]):
                print
                print "***************Starting example " + str(i)
                print ex
                Network = forwardPropogate(ex, Network)
                print "after forwards prop"
                showNetwork(trainingSet, Network, attributes, classifs, False)
                Network, error = backwardPropogate(ex, Network, alpha, mu)
                print "after backwards prop"
                showNetwork(trainingSet, Network, attributes, classifs, False)
                Network = resetNodeInputs(Network) # Inputs (but not weights) need to be reset after each iteration
                trainingSumMSE += error
            
            MSE = trainingSumMSE#/len(trainingSet)
            print
            print "Iteration Mean Squared Error: " + str(MSE)
#             alpha = 1000/(1000 + (j + 2)) # Decrease learning rate
            mu = 1000/(1000 + (j + 2)) # Decrease influence of momentum
        
        #showNetwork(trainingSet, Network, attributes, classifs, False)
        pickle.dump(Network, open('Network.dat', 'w'))
        pickle.dump(testSet, open('testSet.dat', 'w'))

############################################################################################    
############################################## Forward Propogation
# An implementation of forward propagation
def forwardPropogate(dataItem, network):
    
    ### Pass data input values into input layer
    inputNodes = network[0] 
    for i, val in enumerate(dataItem.values):
        input = inputNodes[i]
        input.value = val
    
    # Progress! through rest of layers
    for i in range(1, len(network)):
        network[i] = feedInputsForward(network[i-1], network[i])
        
    return network

# Channels output from each node at previous level into input at next level    
def feedInputsForward(prevLayer, nextLayer):
    for nextNode in nextLayer:
        for prevNode in prevLayer:
            a = prevNode.output() # output = g(weighted sum of inputs to prevNode)
            nextNode.inputs.append(a)        
    return nextLayer
    
############################################################################################    
############################################## Backward Propogation
def backwardPropogate(ex, network, alpha, mu):
    
    # Keep track of squared error to evaluate performance
    sumSquaredError = 0.0
    
    # Update weights between output and last hidden layer
    outNodes = network[-1]
    backLayer = network[-2]
    for out in outNodes:
        print "Out " + out.classif + ": " + str(out.output())

        # Calculate error (error = target - output)
        if ex.classif == out.classif:
            error = 1 - out.output()
        else:
            error = 0 - out.output()
        sumSquaredError += math.pow(error, 2)
        
        # Update weights 
        out.deltaPrev = out.delta # Save gradient from previous iteration -- for momentum calculation
        out.delta = getGradient(out.inputs, out.weights, error)
        out.weights = updateWeights(out, backLayer, alpha, mu)

    MSE = sumSquaredError/len(outNodes)
            
    # Update weights between hidden layers
    for i in range(2, len(network)):
        nextLayer = network[-i + 1]
        currentLayer = network[-i]
        backLayer = network[-i - 1]
        
        for i, node in enumerate(currentLayer):

            # Calculate error by summing up over all nodes in next level
            error = 0
            for nextNode in nextLayer:
                wt = nextNode.weights[i+1] # bypass dummy weight to find wt connecting node w nextNode
                error += wt * nextNode.delta
            
            # Update weights
            node.deltaPrev = node.delta # For momentum calculation
            node.delta = getGradient(node.inputs, node.weights, error)
            node.weights = updateWeights(node, backLayer, alpha, mu)

    return network, MSE

# Calculate gradient values to update weights         
def getGradient(inputs, weights, error):
    wtInputs = weightedInputs(inputs, weights)
    delta = gprime(wtInputs) * error   
    
    return delta 

# Calculate momentum (incorporates information from gradient of previous iteration)
# Helps avoid settling in local minima
def getMomentum(node, mu):
    if node.deltaPrev:
        return mu * node.deltaPrev
    else:
        return 0

# Use learning rate, momentum and gradient to update all weights into a given node
def updateWeights(node, backLayer, alpha, mu):
    weights = node.weights
    
    # Update weight for constant bias term
    weights[0] = (weights[0] + alpha * 1 * node.delta) + getMomentum(node, mu)
    
    # Iterate over all nodes in previous layer connected to current node to update corresponding weight 
    for i in range(1, len(weights)):
        hidden = backLayer[i-1]
        aj = hidden.output()
        weights[i] = weights[i] + alpha * aj * node.delta + getMomentum(node, mu)
    
    return weights
    
############################################################################################    
############################################## Network
def initLayers(attributes, classifs, numHiddenLayers, numNodesPerLayer):
    Network = []
    
    # Initialize input nodes
    inputNodes = []
    for attr in attributes:
        inNode = InputNode(attr)
        inputNodes.append(inNode)
    Network.append(inputNodes)
    
    # Initialize hidden nodes
    for i in range(numHiddenLayers):
        hiddenLayer = []
        numHiddenNodeWts = len(Network[-1]) + 1 # Number nodes in previous level + 1  (include weight for constant bias term)
        for j in range(numNodesPerLayer):
            hidden = HiddenNode(i, numHiddenNodeWts)
            hiddenLayer.append(hidden)
        Network.append(hiddenLayer)
    
    # Initialize output nodes
    outputNodes = []
    numOutNodeWts = numNodesPerLayer + 1 # + 1  -- include weight for constant bias term
    for classif in classifs:
        outNode = OutputNode(classif, numOutNodeWts)
        outputNodes.append(outNode)
    Network.append(outputNodes) 
    
    return Network

# Removes inputs from hidden and output nodes before each forward/backward iteration 
def resetNodeInputs(network):
    for layer in network[1:]:
        for node in layer:
            node.inputs = [1] # Leave only constant bias term
    return network
        
# Prints network nodes and training data, for debugging purposes
def showNetwork(data, network, attributes, classifs, showData):
    print "~~~~~~~~~~~~~~~~~~~~~~~~~~~NETWORK~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    print "Names of attributes: " + str(attributes)
    print "\nTypes of classifications: " + str(classifs)
    print "------"
    print "Input Nodes: "
    for node in network[0]:
        print node
    print "\nHidden Nodes: "
    for i, layer in enumerate(network[1:-1]):
        print "--hidden layer " + str(i)
        for node in layer:
            print node
    print "\nOutput Nodes: "
    for node in network[-1]:
        print node
    print "------"
    if showData:
        print "Data: "
        for item in data:
            print item.formatWithAttrs(attributes)
        
    print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
            
############################################################################################    
############################################## Nodes
# Base 'Node' superclass for HiddenNode and OutputNode subclasses      
class Node:
    def __init__(self, numWeights):
        self.weights = [random.uniform(0.0001, 0.1) for i in range(numWeights)]
        self.inputs = [1] # Start with 1 constant bias term
        self.delta = None
        self.deltaPrev = None
    
    def output(self):
        weightedIn = weightedInputs(self.inputs, self.weights)
        return g(weightedIn)
    
class HiddenNode(Node):
    def __init__(self, level, numWeights):
        Node.__init__(self, numWeights)
        self.l = level
    
    def __str__(self):
        toPrint = "Hidden, level " + str(self.l) + ":\n*****inputs: " + str(self.inputs) + \
                    "\n*****weights: " + str(self.weights) + "\n"
        return toPrint

class OutputNode(Node):
    def __init__(self, classification, numWeights):
        Node.__init__(self, numWeights)
        self.classif = classification
    
    def __str__(self):
        toPrint = "Output, " + self.classif + ":\n*****inputs: " + str(self.inputs) + \
                    "\n*****weights: " + str(self.weights) + \
                    "\n*****output: " + str(self.output()) + "\n"
        return toPrint

######### InputNode doesn't inherit -- has no weights, output is different
# Passes data from examples directly into first hidden layer
class InputNode:
    def __init__(self, attrVal):
        self.attr = attrVal
    
    value = 0
    
    def output(self):
        return self.value
    
    def __str__(self):
        toPrint = "Input, " + self.attr + ":\n*****value: " + str(self.value) + "\n"
        return toPrint

######### Activation function:
#### All hidden and output nodes will use the same activation function
# Weighted sum of inputs
def weightedInputs(inputs, weights):
    sum = 0.0
    for i in range(len(inputs)):
        sum += weights[i]*inputs[i]
    return sum

# Activation function: logistic function
def g(x):
    return 1/(1 + math.exp(-x))

# Derivative of the logistic function
def gprime(x):
    return g(x)*(1 - g(x))

############################################################################################
############################################## Data
# Parse each csv line into a 'DataItem' instance, while keeping track of all classifications
def parseData(data):
    sortedData = defaultdict(list)
    for ex in data:
        classif = ex[0]
        name = ex[1]
        values = [int(val) for val in ex[2:]]
        item = DataItem(name, classif, values)
        sortedData[classif].append(item)
    return sortedData

# For each classification, randomly divide corresponding data items into training and test sets
def separateTestData(sortedData, testSize):
    testSet = []
    trainingSet = []
    for classifSet in sortedData.values():
        random.shuffle(classifSet)
        testSet = testSet + classifSet[:testSize]
        trainingSet = trainingSet + classifSet[testSize:]
    return testSet, trainingSet
    
class DataItem:
    def __init__(self, name, classification, values):
        self.name = name
        self.values = values
        self.classif = classification
        
    def __str__(self):
        toPrint = self.name + " -- " + self.classif 
        return toPrint + "\n"
    
    def formatWithAttrs(self, attributes):
        toPrint = self.name + " -- " + self.classif 
        for i, attr in enumerate(attributes):
            toPrint += "\n**" + attr + " : " + str(self.values[i])
        return toPrint + "\n"


if __name__ == "__main__":
    main()