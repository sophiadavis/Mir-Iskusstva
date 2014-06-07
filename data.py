'''
Sophia Davis
6/7/2014
data.py

Classes and functions for storing/processing the data used to train and test neural network.
'''
from collections import defaultdict
import random
import itertools

# Parse each csv line into a 'DataItem' instance
### Returns dictionary of all data items, sorted by classification
def parseData(data):
    sortedData = defaultdict(list)
    for ex in data:
        classif = ex[0]
        name = ex[1]
        values = [float(val) for val in ex[2:]]
        item = DataItem(name, classif, values)
        sortedData[classif].append(item)
    return sortedData

# For each classification, randomly divide corresponding data items into training and test sets
# Each classification present in the data set will be represented proportionally in 
### each test/training set.
### The value of k should divide evenly into the number of items in all classification sets.
def separateTestData(sortedData, k):
    
    if k == 1:
        raise Exception("k must be greater than 1.")

    testSets = defaultdict(list)
    trainingSets = defaultdict(list)
    
    # Divide each classification set into k test and training groups
    for classifSet in sortedData.values():
        
        testSetSize = int(float(len(classifSet))/k)
        random.shuffle(classifSet)
        
        for i in range(k):
            testSets[i] = testSets[i] + classifSet[(i * testSetSize) : ((i + 1) * testSetSize)]
            trainingSets[i] = trainingSets[i] + classifSet[ : (i * testSetSize)] + classifSet[((i + 1) * testSetSize) : ]
    
    return testSets, trainingSets
    
class DataItem:
    def __init__(self, name, classification, values):
        self.name = name
        self.values = values
        self.classif = classification
        
    def __str__(self):
        toPrint = self.name + " -- " + self.classif 
        return toPrint
    
    def formatWithAttrs(self, attributes):
        toPrint = self.name + " -- " + self.classif 
        for i, attr in enumerate(attributes):
            toPrint += "\n**" + attr + " : " + str(self.values[i])
        return toPrint + "\n"