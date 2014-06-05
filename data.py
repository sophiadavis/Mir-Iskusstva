'''
Sophia Davis
5/31/2014
Data.py

Classes and functions for data items used to train and test neural network
'''
from collections import defaultdict
import random
import itertools

# Parse each csv line into a 'DataItem' instance, while keeping track of all classifications
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
def separateTestData(sortedData, k):
    
    # If no test set is desired
    if k == 1:
        raise Exception("k must be greater than 1.")

    testSets = defaultdict(list)
    trainingSets = defaultdict(list)
    
    for classifSet in sortedData.values():
        
        # Divide current classification set into k test and training groups
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