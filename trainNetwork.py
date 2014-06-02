'''
Sophia Davis
5/31/2014
trainNetwork.py

Implements backpropogation
'''

import sys
import csv
import math

def main():
    if len(sys.argv) < 2:
        sys.stderr.write('Usage: python ' + sys.argv[0] + ' trainingdata.csv\n')
        sys.exit(1)
    else:
        print "Reading in data..."
        with open(sys.argv[1], 'rb') as f:
            imagereader = csv.reader(f)
            data = []
            for row in imagereader:
                data.append(row)
        f.close()
        
        print "Training neural network..."
        
        ######################## Initialization
        ####### Set network properties
        alpha = 0.4
        numHiddenLayers = 1
        numNodesPerLayer = 5
        Network = NeuralNetwork(alpha, numHiddenLayers, numNodesPerLayer)
        global attributes
        global classifs
        
        ####### Parse training data    
        ### Collect all possible attributes and classifications
        attributes = data[0][2:] # first index is the classification
        classifsSet = set([]) 
        
        ### Turn each line into 'DataItem' instance
        data = data[1:]
        trainingSet = []
        for ex in data:
            classif = ex[0]
            name = ex[1]
            values = [int(val) for val in ex[2:]] # add x0
            item = DataItem(name, classif, values)
            trainingSet.append(item)
            classifsSet.add(classif)
        classifs = list(classifsSet)
        
        # Initialize input, hidden, and output layers    
        Network.initLayers()
            
        
        
           
        
        ####################### check!
        showNetwork(trainingSet, Network, True)
        
        # pass input to first layer
        for ex in trainingSet[0:1]:
            Network = forwardPropogate(ex, Network)
        
        ####################### Start ForwardPropogation!
        showNetwork(trainingSet, Network, False)
        
    

############################################################################################    
############################################## Forward Propogation
def forwardPropogate(dataItem, Network):

    net = Network.layers
    ### initialize input values
    inputNodes = net[0] 
    for i, val in enumerate(dataItem.values):
        input = inputNodes[i]
        input.value = val
    
    for i in range(1,len(net)):
        net[i] = feedForward(net[i-1], net[i])
        
    return Network
    
def feedForward(prevLayer, nextLayer):
    for nextNode in nextLayer:
        for prevNode in prevLayer:
            a = prevNode.output()
            nextNode.inputs.append(a)        
    return nextLayer
    
############################################################################################    
############################################## Backward Propogation
def backwardPropogate(Network, alpha):
    pass
    
############################################################################################    
############################################## Network
class NeuralNetwork:
    def __init__(self, alpha, numHiddenLayers, numNodesPerLayer):
        self.alpha = alpha
        self.numHidden = numHiddenLayers
        self.numNodesPerLayer = numNodesPerLayer  
        self.layers = []
    
    def initLayers(self):
        # initialize input nodes
        inputNodes = []
        for attr in attributes:
            inNode = InputNode(attr)
            inputNodes.append(inNode)
        self.layers.append(inputNodes)
        
        # Initialize hidden nodes
        numHiddenNodeWts = len(inputNodes) + 1 # + 1  -- dummy for constant term
        for i in range(self.numHidden):
            hiddenLayer = []
            for j in range(self.numNodesPerLayer):
                hidden = HiddenNode(i, numHiddenNodeWts)
                hiddenLayer.append(hidden)
            self.layers.append(hiddenLayer)
        
        # Initialize output nodes
        outputNodes = []
        numOutNodeWts = self.numNodesPerLayer + 1
        for classif in classifs:
            outNode = OutputNode(classif, numOutNodeWts)
            outputNodes.append(outNode)
        self.layers.append(outputNodes) 
        
# Prints network nodes and training data, for debugging purposes
def showNetwork(data, Network, showData):
    print "~~~~~~~~~~~~~~~~~~~~~~~~~~~NETWORK~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    print "Names of attributes: " + str(attributes)
    print "\nTypes of classifications: " + str(classifs)
    print "------"
    print "Input Nodes: "
    for node in Network.layers[0]:
        print node
    print "\nHidden Nodes: "
    for i, layer in enumerate(Network.layers[1:-1]):
        print "--hidden layer " + str(i)
        for node in layer:
            print node
    print "\nOutput Nodes: "
    for node in Network.layers[-1]:
        print node
    print "------"
    if showData:
        print "Data: "
        for item in data:
            print item
        
    print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
            
############################################################################################    
############################################## Nodes
# Base 'Node' superclass  for HiddenNode and OutputNode subclasses      
class Node:
    def __init__(self, numWeights):
        self.weights = [0.05] * numWeights
        self.inputs = [1] # start with dummy for constant term
    
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
class DataItem:
    def __init__(self, name, classification, values):
        self.name = name
        self.values = values
        self.classif = classification
        
    def __str__(self):
        toPrint = self.name + " -- " + self.classif 
        for i, attr in enumerate(attributes):
            toPrint += "\n**" + attr + " : " + str(self.values[i])
        return toPrint + "\n"


if __name__ == "__main__":
    main()