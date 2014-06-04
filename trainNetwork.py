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
import pickle
import time

from node import *
from data import *

def main():
    if len(sys.argv) < 2:
        sys.stderr.write('Usage: python ' + sys.argv[0] + ' trainingDataFile.csv\n')
        sys.exit(1)
    else:
        timerStart = time.time()
        print "Reading in data...\n"
        with open(sys.argv[1], 'rb') as f:
            imagereader = csv.reader(f)
            data = []
            for row in imagereader:
                data.append(row)
        f.close()
        
        ######################## Initialization
        ####### Set network properties
        alpha = 1000.0/(1000.0 + 1.0)
        mu = alpha/2 # Control parameter for momentum
        testSize = 1 # Size of test set (from group of examples with each classification)
        numHiddenLayers = 1
        numNodesPerLayer = 24
        
        ####### Parse data    
        ### Collect all possible attributes and classifications
        attributes = data[0][2:] # First index is the classification, second is file name
        data = data[1:]
        sortedData = parseData(data) # Dictionary of data items sorted by classif
        classifs = sortedData.keys()

        print "Separating training and test sets..."
        testSet, trainingSet = separateTestData(sortedData, testSize)
        
        print "Training neural network on training data..."
        print "Number hidden layers: " + str(numHiddenLayers)
        print "Nodes per hidden layer: " + str(numNodesPerLayer)
        
        # Initialize input, hidden, and output layers    
        Network = initLayers(attributes, classifs, numHiddenLayers, numNodesPerLayer)
                
        # Forward and Backward propagate, given each data item
        random.shuffle(trainingSet) # Randomize order of data set
        for j in range(50000):
            print
            print "~~~~~~~~~~~~~~~~~Iteration " + str(j)
            trainingSumMSE = 0.0
            for i, ex in enumerate(trainingSet):
                Network = forwardPropogate(ex, Network)
                Network, error, wtChange = backwardPropogate(ex, Network, alpha, mu)
                Network = resetNodeInputs(Network) # Inputs (but not weights) need to be reset after each iteration
                trainingSumMSE += error
                if j == 49999:
                    print
                    print "FINAL ITERATION: ---- "
                    print ex
                    for out in Network[-1]:
                        print "---" + out.classif + ": " + str(out.output())
                    print
            
            MSE = trainingSumMSE/len(trainingSet)
            print
            print "Iteration MSE: " + str(MSE)
            print "Mean weight change: " + str(wtChange)
            alpha = 1000.0/(1000.0 + (j + 2.0)) # Decrease learning rate
            mu = alpha/2
#             mu = 1000.0/(1000.0 + (2*j + 2.0)) # Decrease influence of momentum
            print alpha
            print mu
        
        timerEnd = time.time()
        print "Time elapsed: " + str(float(timerEnd - timerStart)/(60)) + " minutes."
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
    
    # Keep track of squared error and average magnitude of weight changes to evaluate performance
    sumSquaredError = 0.0
    wtChange = 0.0
    count = 0 # For finding mean weight change
    
    # Update weights between output and last hidden layer
    outNodes = network[-1]
    backLayer = network[-2]
    for out in outNodes:

        # Calculate error (error = target - output)
        if ex.classif == out.classif:
            error = 1 - out.output()
        else:
            error = 0 - out.output()
        sumSquaredError += math.pow(error, 2)
        
        # Update weights 
        out.delta = getGradient(out.inputs, out.weights, error)
        
        newWts = updateWeights(out, backLayer, alpha, mu)
        wtChange += meanChange(out.weights, newWts)
        count += 1
        
        out.weights = newWts
        
    MSE = float(sumSquaredError)/len(outNodes)
            
    # Update weights between hidden layers
    for i in range(2, len(network)):
        nextLayer = network[-i + 1]
        currentLayer = network[-i]
        backLayer = network[-i - 1]
        
        for i, node in enumerate(currentLayer):

            # Calculate error by summing up over all nodes in next level
            error = 0.0
            for nextNode in nextLayer:
                wt = nextNode.weights[i+1] # Bypass dummy weight to find weight connecting node w nextNode
                error += wt * nextNode.delta
            
            # Update weights
            node.delta = getGradient(node.inputs, node.weights, error)
            
            newWts = updateWeights(node, backLayer, alpha, mu)
            wtChange += meanChange(node.weights, newWts)
            count += 1
        
            node.weights = newWts
    
    meanWtChange = float(wtChange)/count

    return network, MSE, meanWtChange

# Calculate mean magnitude of the difference between previous and updated weights 
def meanChange(oldWts, newWts):
    change = map(lambda x,y: math.fabs(x - y), oldWts, newWts)
    return float(sum(change))/len(change)

# Calculate gradient values to update weights         
def getGradient(inputs, weights, error):
    wtInputs = weightedInputs(inputs, weights)
    delta = gprime(wtInputs) * error   
    
    return delta 

# Use learning rate, momentum and gradient to update all weights into a given node
# Momentum = fraction of weight update given previous data item
### Helps avoid settling in local minima
def updateWeights(node, backLayer, alpha, mu):
    wts = node.weights
    updatedWts = []
    
    # Update weight for constant bias term
    updateTerm = alpha * 1 * node.delta + mu * node.prevWtUpdates[0]
    newWt0 = wts[0] + updateTerm
    
    updatedWts.append(newWt0)
    node.prevWtUpdates[0] = updateTerm
    
    # Iterate over all nodes in previous layer connected to current node to update corresponding weight 
    for i in range(1, len(wts)):
        hidden = backLayer[i-1]
        aj = hidden.output()
        
        updateTerm = (alpha * aj * node.delta) + (mu * node.prevWtUpdates[i])
        newWt = wts[i] + updateTerm
        
        updatedWts.append(newWt)
        node.prevWtUpdates[i] = updateTerm
    
    return updatedWts
    
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


if __name__ == "__main__":
    main()