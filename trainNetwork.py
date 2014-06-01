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
        print "Training neural network..."
        
        ######################## initialization
        # set network properties
        alpha = 0.4
        numHiddenLayers = 1
        numNodesPerLayer = 5
        global attributes
        global classifs

        # read in image color info
        with open(sys.argv[1], 'rb') as f:
            imagereader = csv.reader(f)
            data = []
            for row in imagereader:
                data.append(row)
        f.close()
    
        # initialize input nodes
        attributes = data[0][2:] # first index is the classification
        inputNodes = []
        for attr in attributes:
            inNode = InputNode(attr)
            inputNodes.append(inNode)
    
        # initialize hidden nodes
        hiddenNodes = []
        for i in range(numHiddenLayers):
            for j in range(numNodesPerLayer):
                hidden = HiddenNode(i)
                hiddenNodes.append(hidden)
        
        # parse training data
        ### collect all possible classifications
        classifsSet = set([]) 
        
        data = data[1:]
        trainingSet = []
        for ex in data:
            classif = ex[0]
            name = ex[1]
            values = [int(val) for val in ex[2:]]
            item = DataItem(name, classif, values)
            trainingSet.append(item)
            classifsSet.add(classif)
        
        # initialize output nodes
        classifs = list(classifsSet)
        outputNodes = []
        for classif in classifs:
            outNode = OutputNode(classif)
            outputNodes.append(outNode)
            
        ####################### check!
        showNetwork(trainingSet, inputNodes, hiddenNodes, outputNodes)
        
        ######################## continue!
def showNetwork(data, inputs, hiddens, outputs):
    print "~~~~~~~~~~~~~~~~~~~~~~~~~~~NETWORK~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    print "Names of attributes: " + str(attributes)
    print "\nTypes of classifications: " + str(classifs)
    print "------"
    print "Input Nodes: "
    for node in inputs:
        print node
    print "\nHidden Nodes: "
    for node in hiddens:
        print node
    print "\nOutput Nodes: "
    for node in outputs:
        print node
    print "------"
    print "Data: "
    for item in data:
        print item
        
    print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
############################################## nodes        
class Node:
    def __init__(self):
        self.weights = [0.05, 0.05, 0.05]
        self.inputs = []
    
    def output(self):
        g(self.inputs, self.weights)

class InputNode(Node):
    def __init__(self, attrVal):
        Node.__init__(self)
        self.attr = attrVal
    
    def __str__(self):
        toPrint = "Input, " + self.attr + ":\n*****inputs: " + str(self.inputs) + \
                    "\n*****weights: " + str(self.weights) + \
                    "\n*****output: " + str(self.output) + "\n"
        return toPrint
    
class HiddenNode(Node):
    def __init__(self, level):
        Node.__init__(self)
        self.l = level
    
    def __str__(self):
        toPrint = "Hidden, level " + str(self.l) + ":\n*****inputs: " + str(self.inputs) + \
                    "\n*****weights: " + str(self.weights) + \
                    "\n*****output: " + str(self.output) + "\n"
        return toPrint

class OutputNode(Node):
    def __init__(self, classification):
        Node.__init__(self)
        self.classif = classification
    
    def __str__(self):
        toPrint = "Output, " + self.classif + ":\n*****inputs: " + str(self.inputs) + \
                    "\n*****weights: " + str(self.weights) + \
                    "\n*****output: " + str(self.output) + "\n"
        return toPrint

def g(inputs, weights):
    sum = 0.0
    for i in range(len(inputs)):
        sum += weights[i]*inputs[i]
    return logistic(sum)

def logistic(x):
    return 1/(1 + math.exp(-x))
    
############################################## data
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