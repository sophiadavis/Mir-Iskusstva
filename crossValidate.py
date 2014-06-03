'''
Sophia Davis
5/31/2014
classify.py

Cross validates neural network given test data set
'''
import sys
import pickle
import copy
from trainNetwork import *

def main():
    if len(sys.argv) < 2:
        sys.stderr.write('Usage: python ' + sys.argv[0] + ' trainingDataFile.csv\n')
        sys.exit(1)
    else:
        print "\nProcessing network..."
        Network = pickle.load(open(sys.argv[1], 'r'))
        testSet = pickle.load(open(sys.argv[2], 'r'))

        testSumMSE = 0.0
        for item in testSet:
            trained = copy.deepcopy(Network)
            output = forwardPropogate(item, trained)
            outNodes = output[-1]
            testSumMSE += getOutputError(item, outNodes, True)
        MSE = testSumMSE/len(testSet)
        print "\n************************"
        print "MSE on test set: " + str(MSE)
        print
            
# Calculate network performance classifying test data item
# If show == true, displays all network output predictions
def getOutputError(ex, outNodes, show):
    sumSquaredError = 0.0
    if show:
        print
        print ex
    
    for out in outNodes:
        if show:
            print "---" + out.classif + ": " + str(out.output())
        
        # Calculate error (error = target - output)
        if ex.classif == out.classif:
            error = 1 - out.output()
        else:
            error = 0 - out.output()
        sumSquaredError += math.pow(error, 2)
        
    MSE = sumSquaredError/len(outNodes)
    return MSE

if __name__ == "__main__":
    main()