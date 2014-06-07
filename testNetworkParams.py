'''
Sophia Davis
6/7/2014
testNetworkParams.py

Trains network on various combinations of network parameters given data in csv format
    (specified in 1st command line argument).
Information about performance on training set (MSE from each iteration and final 
    average weight change) is saved into csv file (2nd command line argument).
'''
import sys
import csv
import itertools

from data import *
from trainNetwork import *

def main():
    if len(sys.argv) < 3:
        sys.stderr.write('Usage: python ' + sys.argv[0] + ' trainingDataFile.csv outputFile.csv\n')
        sys.exit(1)
    else:
        print "Reading in data...\n"
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
        
        # Add all data to training set (no test set)
        dataSet = list(itertools.chain.from_iterable(sortedData.values()))
        
        iterations = 10
        paramCombos = []
        printCombos = []
        
        ####### Prepare csv file to store results
        csvName = sys.argv[2]
        with open(csvName, 'wb') as f:
            writer = csv.writer(f)
            writer.writerow(["Alpha", "Mu", "Structure", "FinalAvgWtChange"] + range(1, iterations + 1))
        f.close()
        
        ####### Set combinations of network properties
        
        # Best performing
        learningRate = lambda x: 1000.0/(1000.0 + x)
        momentumRate = lambda x: learningRate(x)/2
        numNodesPerLayer = [24] # [nodesInLayer0, nodesInLayer1, nodesInLayer2 ...
        paramCombos.append([learningRate, momentumRate, numNodesPerLayer])
        printCombos.append(["1000.0/(1000.0 + x)", "alpha/2", str(numNodesPerLayer)])
        
        learningRate = lambda x: 1000.0/(1000.0 + x)
        momentumRate = lambda x: learningRate(x)/2
        numNodesPerLayer = [44] # [nodesInLayer0, nodesInLayer1, nodesInLayer2 ...
        paramCombos.append([learningRate, momentumRate, numNodesPerLayer])
        printCombos.append(["1000.0/(1000.0 + x)", "alpha/2", str(numNodesPerLayer)])
        
        learningRate = lambda x: 1000.0/(1000.0 + x)
        momentumRate = lambda x: learningRate(x)/2
        numNodesPerLayer = [32] # [nodesInLayer0, nodesInLayer1, nodesInLayer2 ...
        paramCombos.append([learningRate, momentumRate, numNodesPerLayer])
        printCombos.append(["1000.0/(1000.0 + x)", "alpha/2", str(numNodesPerLayer)])
        
        learningRate = lambda x: 1000.0/(1000.0 + x)
        momentumRate = lambda x: learningRate(x)/2
        numNodesPerLayer = [24, 32] # [nodesInLayer0, nodesInLayer1, nodesInLayer2 ...
        paramCombos.append([learningRate, momentumRate, numNodesPerLayer])
        printCombos.append(["1000.0/(1000.0 + x)", "alpha/2", str(numNodesPerLayer)])
        
        # ...
        
        ####### For each parameter combination, train on entire data set, save results to csv
        for i in range(len(paramCombos)):
            print "\nTraining neural network on parameter combination " + str(i + 1) + "..."
            Network, allMSEs, wtChange = trainNetwork(dataSet, attributes, classifs, paramCombos[i][0], paramCombos[i][1], paramCombos[i][2], iterations, True)
            with open(csvName, 'a') as f:
                writer = csv.writer(f)
                writer.writerow(printCombos[i] + [wtChange] + allMSEs)
            f.close()   
            
if __name__ == "__main__":
    main()          
            
            
            
            