'''
Sophia Davis
5/31/2014
separateData.py

Separated csv file into training and test sets for use in k-fold cross-validation.
Each classification present in the data set will be represented proportionally in 
    each test/training set (unless the value of k does not divide evenly into the 
    number of items in all classification sets).
'''

import sys
import csv
import random
import pickle
from collections import defaultdict

from data import *

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
        
        ####### Parse data    
        ### Collect all possible attributes and classifications
        attributes = data[0][2:] # First index is the classification, second is file name
        data = data[1:]
        sortedData = parseData(data) # Dictionary of data items sorted by classif
        classifs = sortedData.keys()

        print "Separating training and test sets..."
        k = 20 # number of test/training sets
        testSets, trainingSets = separateTestData2(sortedData, k)
        for i in range(len(testSets)):
            print
            print "TEST AND TRAINING SETS " + str(i)
            print "Test set:" + str(i)
            print len(testSets[i])
            print testSets[i][0]
            print trainingSets[i].count(testSets[i][0])
            for item in testSets[i]:
                print trainingSets[i].count(item)
                
            print
            print "Training set:" + str(i)
            print len(trainingSets[i])
            for item in trainingSets[i]:
                print item

# For each classification, randomly divide corresponding data items into training and test sets
def separateTestData2(sortedData, k):
    testSets = defaultdict(list)
    trainingSets = defaultdict(list)
    
    for classifSet in sortedData.values():
        
        # Divide current classification set into k test and training groups
        testSize = int(float(len(classifSet))/k)
        random.shuffle(classifSet)
        
        for i in range(k):
            testSets[i] = testSets[i] + classifSet[(i * testSize) : ((i + 1) * testSize)]
            trainingSets[i] = trainingSets[i] + classifSet[ : (i * testSize)] + classifSet[((i + 1) * testSize) : ]
    
    return testSets, trainingSets
    

if __name__ == "__main__":
    main()    
    