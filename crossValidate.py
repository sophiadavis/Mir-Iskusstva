'''
Sophia Davis
5/31/2014
classify.py

Uses k-fold cross validation to train and test neural network
Separates csv file into training and test sets for use in k-fold cross-validation.
Each classification present in the data set will be represented proportionally in 
    each test/training set (unless the value of k does not divide evenly into the 
    number of items in all classification sets).
'''
import sys
import pickle
import copy

from trainNetwork import *
from node import *
from data import *

def main():
    if len(sys.argv) < 2:
        sys.stderr.write('Usage: python ' + sys.argv[0] + ' trainingDataFile.csv\n')
        sys.exit(1)
    else:
        print "\nReading in data..."
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

        print "...Separating training and test sets..."
        k = 20 # Number of test/training sets
        testSets, trainingSets = separateTestData(sortedData, k)
                    
        print "...Training and test sets complete.\n"
        
        ####### Set network parameters
        learningRate = lambda x: 1000.0/(1000.0 + x)
        momentumRate = lambda x: learningRate(x)/2
        numNodesPerLayer = [24] # [nodesInLayer0, nodesInLayer1, nodesInLayer2 ...]
        
        ####### Train and test networks
        for i in range(k):
            print "Training network, round " + str(i) + " of " + str(k) + "."
            Network = trainNetwork(trainingSets[i], attributes, classifs, learningRate, momentumRate, numNodesPerLayer, False)
            
            print "Cross-validating network, round " + str(i) + " of " + str(k) + "."
            testSetSumMSE = 0.0
            for item in testSets[i]:
                trained = copy.deepcopy(Network)
                output = forwardPropogate(item, trained)
                outNodes = output[-1]
                error, predictions = getOutputError(item, outNodes, True)
                testSetSumMSE += error
                print "Max prediction: " + predictions[max(predictions.keys())]
            MSE = testSetSumMSE/len(testSets[i])
            print "\n************************"
            print "MSE on test set: " + str(MSE)
            print
            
# Calculate network performance classifying test data item
# Return MSE and predictions (output of all nodes)
# If show == True, displays all network output predictions
def getOutputError(ex, outNodes, show):
    sumSquaredError = 0.0
    predictions = {}
    
    if show:
        print
        print ex
    
    for out in outNodes:
        if show:
            print "---" + out.classif + ": " + str(out.output())
        
        predictions[out.output()] = out.classif
        
        # Calculate error (error = target - output)
        if ex.classif == out.classif:
            error = 1 - out.output()
        else:
            error = 0 - out.output()
        sumSquaredError += math.pow(error, 2)
        
    MSE = sumSquaredError/len(outNodes)
    return MSE, predictions

if __name__ == "__main__":
    main()