'''
Sophia Davis
5/31/2014
testNetworkParams.py

Trains network on various combinations of network parameters, saving MSE for each iteration,
    for comparison.
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
        sortedData = parseData(data) # Dictionary of data items sorted by classif
        classifs = sortedData.keys()

        dataSet = list(itertools.chain.from_iterable(sortedData.values()))
        
        ####### Prepare csv file to store results
        csvName = sys.argv[2]
        with open(csvName, 'wb') as f:
            writer = csv.writer(f)
            writer.writerow(["Alpha", "Mu", "Structure", "FinalAvgWtChange"] + range(1, 11))
        f.close()
        
        ####### Set combinations of network properties
        
        iterations = 10
        paramCombos = []
        printCombos = []
        
        learningRate = lambda x: 1000.0/(1000.0 + x)
        momentumRate = lambda x: 0.9
        numNodesPerLayer = [24] # [nodesInLayer0, nodesInLayer1, nodesInLayer2 ...
        paramCombos.append([learningRate, momentumRate, numNodesPerLayer])
        printCombos.append(["1000.0/(1000.0 + x)", "0.9", str(numNodesPerLayer)])
        
        learningRate = lambda x: 1000.0/(1000.0 + x)
        momentumRate = lambda x: learningRate(x)/2
        numNodesPerLayer = [24] # [nodesInLayer0, nodesInLayer1, nodesInLayer2 ...
        paramCombos.append([learningRate, momentumRate, numNodesPerLayer])
        printCombos.append(["1000.0/(1000.0 + x)", "alpha/2", str(numNodesPerLayer)])
        
        learningRate = lambda x: 0.4
        momentumRate = lambda x: 0.9
        numNodesPerLayer = [24] # [nodesInLayer0, nodesInLayer1, nodesInLayer2 ...
        paramCombos.append([learningRate, momentumRate, numNodesPerLayer])
        printCombos.append(["0.4", "0.9", str(numNodesPerLayer)])
        
        learningRate = lambda x: 1000.0/(1000.0 + x)
        momentumRate = lambda x: 0.9
        numNodesPerLayer = [32] # [nodesInLayer0, nodesInLayer1, nodesInLayer2 ...
        paramCombos.append([learningRate, momentumRate, numNodesPerLayer])
        printCombos.append(["1000.0/(1000.0 + x)", "0.9", str(numNodesPerLayer)])
        
        learningRate = lambda x: 1000.0/(1000.0 + x)
        momentumRate = lambda x: 0.9
        numNodesPerLayer = [32, 24] # [nodesInLayer0, nodesInLayer1, nodesInLayer2 ...
        paramCombos.append([learningRate, momentumRate, numNodesPerLayer])
        printCombos.append(["1000.0/(1000.0 + x)", "0.9", str(numNodesPerLayer)])
        
        learningRate = lambda x: 1000.0/(1000.0 + x)
        momentumRate = lambda x: 0.9
        numNodesPerLayer = [44] # [nodesInLayer0, nodesInLayer1, nodesInLayer2 ...
        paramCombos.append([learningRate, momentumRate, numNodesPerLayer])
        printCombos.append(["1000.0/(1000.0 + x)", "0.9", str(numNodesPerLayer)])
        
        learningRate = lambda x: 1000.0/(1000.0 + x)
        momentumRate = lambda x: 0.9
        numNodesPerLayer = [5] # [nodesInLayer0, nodesInLayer1, nodesInLayer2 ...
        paramCombos.append([learningRate, momentumRate, numNodesPerLayer])
        printCombos.append(["1000.0/(1000.0 + x)", "0.9", str(numNodesPerLayer)])
        
        learningRate = lambda x: 1000.0/(1000.0 + x)
        momentumRate = lambda x: 1.0 - 3.0/(x + 5.0)
        numNodesPerLayer = [24] # [nodesInLayer0, nodesInLayer1, nodesInLayer2 ...
        paramCombos.append([learningRate, momentumRate, numNodesPerLayer])
        printCombos.append(["1000.0/(1000.0 + x)", "1.0 - 3.0/(x + 5.0)", str(numNodesPerLayer)])
        
        learningRate = lambda x: 2 * momentumRate(x)
        momentumRate = lambda x: 1000.0/(2*(1000.0 + x))
        numNodesPerLayer = [24] # [nodesInLayer0, nodesInLayer1, nodesInLayer2 ...
        paramCombos.append([learningRate, momentumRate, numNodesPerLayer])
        printCombos.append(["2*mu", "1000.0/(2*(1000.0 + x))", str(numNodesPerLayer)])
        
        learningRate = lambda x: 1000.0/(2 * (1000.0 + x))
        momentumRate = lambda x: 0.9
        numNodesPerLayer = [24] # [nodesInLayer0, nodesInLayer1, nodesInLayer2 ...
        paramCombos.append([learningRate, momentumRate, numNodesPerLayer])
        printCombos.append(["1000.0/(2 * (1000.0 + x))", "0.9", str(numNodesPerLayer)])
        
        ####### For each parameter combination, train on entire data set, save results to csv
        for i in range(len(paramCombos)):
            print "\nTraining neural network on parameter combination " + str(i + 1) + "..."
            Network, allMSEs, wtChange = trainNetwork(dataSet, attributes, classifs, paramCombos[i][0], paramCombos[i][1], paramCombos[i][2], iterations, True)
            print allMSEs
            with open(csvName, 'a') as f:
                writer = csv.writer(f)
                writer.writerow(printCombos[i] + [wtChange] + allMSEs)
            f.close()   
            
if __name__ == "__main__":
    main()          
            
            
            
            