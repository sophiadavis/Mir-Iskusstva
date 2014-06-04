'''
Sophia Davis
5/31/2014
Data.py

Classes and functions for data items used to train and test neural network
'''
from collections import defaultdict
import random

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
        return toPrint
    
    def formatWithAttrs(self, attributes):
        toPrint = self.name + " -- " + self.classif 
        for i, attr in enumerate(attributes):
            toPrint += "\n**" + attr + " : " + str(self.values[i])
        return toPrint + "\n"