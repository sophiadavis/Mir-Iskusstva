'''
Sophia Davis
5/31/2014
classify.py

Classifies a jpg file, given trained network
'''

import pickle

def main():
    pickle.load(open(sys.argv[2], 'r'))

if __name__ == "__main__":
    main()