'''
Sophia Davis
6/7/2014
crossValidate.py

Uses k-fold cross validation to evaluate the performance of a neural network. 
Data (in csv format - file given in 1st command line argument) is separated into k pairs of
    training and test sets (which are saved in pickle files -- just in case).
Results from each training/test set are saved to a csv file (specified by the 2nd command 
    line argument).
'''
import sys
import copy
import csv
import pickle

from trainNetwork import *
from data import *

def main():
    if len(sys.argv) < 3:
        sys.stderr.write('Usage: python ' + sys.argv[0] + ' dataFile.csv resultsFile.csv\n')
        sys.exit(1)
    else:
        print "\nReading in data..."
        data = []
        with open(sys.argv[1], 'rb') as f:
            imagereader = csv.reader(f)
            for row in imagereader:
                data.append(row)
        f.close()
        
        ####### Parse data    
        ### Collect all possible attributes and classifications
        attributes = data[0][2:] # First index is the classification, second is file name
        data = data[1:]
        sortedData = parseData(data) # Dictionary of data items sorted by classification
        classifs = sortedData.keys()
        
        ####### Separate training and test sets
        print "...Separating training and test sets..."
        k = 2 # Number of test/training sets (must be greater than 1)
        testSets, trainingSets = separateTestData(sortedData, k)
        pickle.dump(testSets, open('testSets.dat', 'w'))
        pickle.dump(trainingSets, open('trainingSets.dat', 'w'))
                    
        print "...Training and test sets complete.\n"
        
        ####### Set network parameters
        learningRate = lambda x: 1000.0/(1000.0 + x)
        momentumRate = lambda x: learningRate(x)/2
        numNodesPerLayer = [24] # [nodesInLayer0, nodesInLayer1, nodesInLayer2 ...]
        iterations = 10
        
        ####### Prepare csv file to store results
        csvName = sys.argv[2]
        paramsList = ["1000.0/(1000.0 + x)", "alpha/2", str(numNodesPerLayer), str(iterations)]
        with open(csvName, 'wb') as f:
            writer = csv.writer(f)
            writer.writerow(["Movement", "File", "MaxPred"] + classifs + ["Alpha", "Mu", "Structure", "Iterations", "FinalAvgWtChange", "InitTrainingSetMSE", "TrainingSetMSE", "TestSetMSE"])
        f.close()
        
        ####### Train and test networks
        for i in range(k):
            csvRows = []
            
            print "Training network, round " + str(i + 1) + " of " + str(k) + "."
            Network, initMSE, trainingMSE, wtChange = trainNetwork(trainingSets[i], attributes, classifs, learningRate, momentumRate, numNodesPerLayer, iterations, False)
            
            print "Cross-validating network, round " + str(i + 1) + " of " + str(k) + "."
            testSetSumMSE = 0.0
            for item in testSets[i]:
            
                trained = copy.deepcopy(Network)
                output = forwardPropogate(item, trained)
                outNodes = output[-1]
                
                error, predictions = getOutputError(item, outNodes, True)
                testSetSumMSE += error
                
                # Find and print classification with maximum predicted value
                maxPred = classifs[predictions.index(max(predictions))] # Predictions are always returned in original order of classifications
                print "Max prediction: " + maxPred
                
                csvRows.append([item.classif, item.name, maxPred] + predictions + paramsList + [wtChange, initMSE, trainingMSE])
            
            MSE = testSetSumMSE/len(testSets[i])
            
            csvRowsMSE = map(lambda x: x + [MSE], csvRows)
            
            with open(csvName, 'a') as f:
                writer = csv.writer(f)
                writer.writerows(csvRowsMSE)
            f.close()
            print "\n************************"
            print "MSE on test set: " + str(MSE)
            print
            
# Calculate and print network performance on data from test set
### Return MSE and prediction output of all nodes
def getOutputError(ex, outNodes, show):
    sumSquaredError = 0.0
    predictions = []
    
    if show:
        print
        print ex
    
    for out in outNodes:
        if show:
            print "---" + out.classif + ": " + str(out.output())
        
        predictions.append(out.output())
        
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