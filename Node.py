'''
Sophia Davis
5/31/2014
Node.py

Classes and activation functions (etc) for nodes in neural network
'''
import random
import math

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