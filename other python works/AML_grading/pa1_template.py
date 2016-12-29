#! /usr/bin/python
# Python Version Used:_
# Fill the above blank
# Example:
# Python Version Used:3
# Python Version Used:2

'''
This is a template outlining the functions we are expecting for us to be able to
interface with an call your code. This is not all of the functions you need.

Note: It is mandatory to manage all your code in this file.
'''

'''
non-native libraries to python you may need you can also use all the libraries
available on silo sever of the university which don't make the code trivial
Example: you cannot call a method for entropy calculation
It is also not required to use all the below libraries.
'''
import numpy as np;
from math import log
from operator import itemgetter
from collections import Counter
import pandas as pd
import sys, os;

'''
Function: load_and_split_data(datapath)
datapath: (String) the location of the data set directory in memory.

This function loads the data and if required you can transform your data as per the requirements

returns: train, test

'''
def load_data(datapath):
    pass

'''
implement a data structure that stores the tree
key's to build
constructor: That creates the Node
setter functions: sets the children to the current node by
                  assigning data and current node as parent to children
getter functions: returns the children/parent as per requirement
'''
class Node:

    def __init__(self):
        '''
        can also be used store all the relevant information to
        a particular node.
        '''
        pass

    # TODO: setter functions

    # TODO: getter functions


'''
Function: train(parameters)
Builds the tree by making a binary split at each node.
If a binary split is not possible it checks for the leaf node condition
and converts the node into leaf node if the leaf node condition is satisfied.

Note: All other function for entropy calculations etc. should be implemented
      without the support of math libraries

train using the training set
'''
def train(depth, train):
    '''
    :param depth: max allowed depth of the tree
    :param train: train data
    :return: Decision Tree
    '''
    pass

'''
Function: test
recursively moves the test data point inside and decides the class label
depending on the leaf node the data point falls into
'''
def test(test, tree):
    '''
    :param test: test data
    :param tree: decision tree
    :return: Accuracy
    '''
    pass

'''
Function: print_dtree
This function prints the decision tree.

Note: It is not required to call this function. However it is should built
      and properly tested. Please use neat printing methods for readability
'''
def print_dtree(tree):
    '''
    :param tree: tree built using the train data
    :return: None
    '''
    pass

'''
Function: print_confusionmatrix
This function let you print the confusion matrix. The parameters
passed to this function can be adjusted as per the need

Note: It is not required to call this function. However it is should built
      and properly tested. Please use neat printing methods for readability
'''
def print_confusionmatrix(test_data, cmtdepth, dtree):
    '''
    make sure you calculate all the cells in the confusion matrix and
    properly assign them to respective values
    :param test_data:
    :param cmtdepth: loop over this variable
    :param dtree: decision tree
    :return: None
    '''
    for depth in range(cmtdepth):
        '''
        all your calculations here
        '''
        TP,FP,TN,FN = 0,0,0,0
        # report confusion matrix
        print(TP)
        print(FP)
        print(TN)
        print(FN)

if __name__ == "__main__":
    # The arguments to your file will be of the following form:
    # <tree_depth> <train_set_path>
    # Ex. 6 2 "train/agaricus-lepiota.data"

    # Get the depth of the tree
    tdepth = sys.argv[1];
    # Get the depth upto which the confusin matrix should be printed
    cmtdepth = sys.argv[2];
    # Get the location of the data set
    datapath = sys.argv[3];


    train_data, test_data = load_data(datapath)
    dtree = train(depth=tdepth, train=train_data)
    '''
    The function call below should be commented before submitting
    '''
    # print_dtree(dtree)

    accuracy = test(test=test_data, tree=dtree)

    '''
    The function call below should be commented before submitting.
    '''

    # Only print statement in the entire code
    print(accuracy)
    print_confusionmatrix(test_data, cmtdepth, dtree)