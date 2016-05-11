"""
1.Using the UCI data sets to implement the Aprori priniciple
************Libraries Used*****************************************
2.Using pickling and unpickling feature of python to save processing time
3.Using numpy and pandas for handling the data
4.Using operator library to sort according to the requirement
"""

import pandas as pd
import pickle
import numpy as np
import operator
import sys
import os
import time
from itertools import combinations

#reading pickled data from disk
def read_data_pickle(dataPath,name):
    with open(dataPath+name+"_pickle", "rb") as foo:
        unpickledData = pickle.load(foo)
    return unpickledData

#pickling data to disk
def write_data_pickle(dataPath,DataToPickle,name):
    with open(dataPath+name+"_pickle", "wb") as foo:
        pickle.dump(DataToPickle,foo)

#transaction list(pandas dataframe) generation as Zeros and Ones as mentioned in the question
def transaction_list_gen(dataPath,fileName):
    with open(dataPath+fileName+"PrepocessedData.txt") as transactions:
    #with open(dataPath+fileName) as transactions:
        transactionsDetails = []
        itemsList = []
        for row in transactions.read().splitlines():
            transaction = row.split(',')
            itemsList += transaction
            transactionsDetails.append(sorted(transaction))
        itemsSet = sorted(set(itemsList))
        itemSetLen = len(itemsSet)
        df = pd.DataFrame(columns=itemsSet)
        for i,transaction in zip(range(len(transactionsDetails)),transactionsDetails):
            transactionNumpyArray = np.zeros(itemSetLen)
            for item in transaction:
                transactionNumpyArray[itemsSet.index(item)] = 1
            df.loc[i] = transactionNumpyArray
    return df

# #function to pickle the data into the disk
# write_data_pickle(transaction_list_gen("groceries.csv"),"matrixCreation")
# #function to read pickled data from disk
# transactionDetails = read_data_pickle("matrixCreation")
# print(transactionDetails.head())


#Function to create frequent Item sets
def freq_item_set_gen(itemSetsFreq,minSup,transactionDetails):
    freqDict = []
    for itemListFreq in itemSetsFreq:
        frequency = transactionDetails.reindex(columns=itemListFreq[0]).prod(axis=1).sum()
        if frequency >= int(minSup):
            freqDict.append([itemListFreq[0],frequency])
    #orderedFreqDict =  sorted(freqDict, key=operator.itemgetter(1),reverse=True)
    #return orderedFreqDict
    return freqDict

# Function to create multi item sets(This function implements f(k-1) * f(k-1) to generate f(k) multi item list)
def multi_item_list_gen(itemsListFreq):
    freqItemsList = [itemsList[0] for itemsList in itemsListFreq]
    extendedItemsList = []
    if freqItemsList != []:
        if len(freqItemsList[0]) >=2:
            for items in freqItemsList:
                for i in range(freqItemsList.index(items)+1,len(freqItemsList)):
                    if items[:-1] == freqItemsList[i][:-1]:                                 #comparing various f(k-1)*f(k-1) to generate fk term
                        extendedItemsList.append([sorted(items+[freqItemsList[i][-1]]),0])
        elif len(freqItemsList[0]) == 1:
            extendedItemsList = multi_item_list_gen1(itemsListFreq,itemsListFreq)
    return extendedItemsList

# Function to create multi item sets(This function implements f(k-1) * f(k1) to generate f(k) multi item list)
def multi_item_list_gen1(itemsListFreq,freqOneItemList):
    freqItemList = sorted([itemsListFreq[0] for itemsListFreq in itemsListFreq])
    extendedItemsList = []
    for itemList in freqItemList:                            #loops only if there is an F(k-1) list exists
        for item in sorted(freqOneItemList):
            itemsCombination = sorted(itemList+item[0])
            # print(itemList)
            # print(item[0][0])
            # print(item[0][0] not in itemList)
            if itemList[-1] < item[0][0] and [itemsCombination,0] not in extendedItemsList:                    #using lexicographical order to prune reoccuring itemsets
                # print(sorted(itemList+item[0]))
                extendedItemsList.append([itemsCombination,0])
    return extendedItemsList

#Output generation-- Prints a the number of itemsets along with a small sample
def printer(candidateItemList,freqItemList,itemListLen):
    # if os.path.isfile(dataPath+"assRules1Output.txt") == True:
    #     os.remove(dataPath+"assRules1Output.txt")
    # sys.stdout = open(dataPath+"assRules1Output.txt","w")
    print("*********Number of items when the list length is ",itemListLen+1,"*********")
    #print("Candidate item lists: ",candidateItemList[0:2],"........",candidateItemList[-2:])
    print("Count of Candidate item lists: ",len(candidateItemList))
    #print("Frequent item lists: ",freqItemList[0:2],"........",freqItemList[-2:])
    print("Count of Frequent item lists: ",len(freqItemList))

#Output generation-- Prints a only the number of itemsets
def printer1(candidateItemList,freqItemList,itemListLen):
    print("*********Number of items when the list length is ",itemListLen+1,"*********")
    #print("Candidate item lists: ",candidateItemList)#[0:2],"........",candidateItemList[-2:])
    print("Count of Candidate item lists: ",len(candidateItemList))
    #print("Frequent item lists: ",freqItemList)#,"........",freqItemList[-2:])
    print("Count of Frequent item lists: ",len(freqItemList))


# Function to make a list of columns that are passed to the freq_item_set_gen function to get the freq of each item
def item_freq_counter(transactionDetails,method,minSup):
    candidateItemList = []
    freqItemList = []
    itemListLen = 0
    itemList = [[[transactionDetails.columns.values[i]],0] for i in range(len(transactionDetails.columns.values))]
    candidateItemList.append(itemList)
    freqItemList.append(itemList)
    candidateItemList[itemListLen] = freq_item_set_gen(candidateItemList[itemListLen],0,transactionDetails)                           #minSup = 0(frequent item generation)
    freqItemList[itemListLen] = freq_item_set_gen(freqItemList[itemListLen],minSup,transactionDetails)    #minSup = user input value
    #printer1(candidateItemList[itemListLen],freqItemList[itemListLen],0)
    #for itemListLen in range(1,itemSetLen):
    while True:
        itemListLen += 1
        if method == 1:
            itemList = multi_item_list_gen1(freqItemList[itemListLen-1],candidateItemList[0])
        elif method == 2:
            itemList = multi_item_list_gen(freqItemList[itemListLen-1])
        candidateItemList.append(itemList)
        freqItemList.append(itemList)
        candidateItemList[itemListLen] = freq_item_set_gen(candidateItemList[itemListLen],0,transactionDetails)                              #minSup = 0(frequent item generation)
        freqItemList[itemListLen] = freq_item_set_gen(freqItemList[itemListLen],minSup,transactionDetails)       #minSup = user input value
        #printer1(candidateItemList[itemListLen],freqItemList[itemListLen],itemListLen)
        if len(candidateItemList[itemListLen]) == 0:
            break
    return candidateItemList,freqItemList

#function to generate the preprocessed file
def preprocessor(dataPath,fileName):
    df = pd.DataFrame.from_csv(dataPath+fileName,index_col=False)
    for column in df.columns.values:
        df[column] = column + df[column]
    #print(df)
    df.to_csv(dataPath+fileName+"PrepocessedData.txt",index=False,header=False)

#this function takes in an itemset and returns all possible combinations of the itemset except null set and the itemset itself
def combinationsGen(itemSet):
    combinationsGenerated = []
    combinationsGeneratedSorted = []
    for length in range(1,len(itemSet)):
        #print(combinations(itemSet,length))
        combinationsGenerated += [list(subset) for subset in combinations(itemSet,length)]
    combinationsGenMaxIndex = len(combinationsGenerated)-1
    for index in range(combinationsGenMaxIndex,-1,-1):
        combinationsGeneratedSorted.append(combinationsGenerated[index])
    return combinationsGeneratedSorted


def itemSetFreq(freqItemList):
    searchSpaceItemSets = []
    searchSpaceItemSetsCount = []
    for level in range(len(freqItemList)):
        for itemSet in freqItemList[level]:
            searchSpaceItemSets.append(itemSet[0])
            searchSpaceItemSetsCount.append(itemSet[1])
    return searchSpaceItemSets,searchSpaceItemSetsCount

#association rule generator to generate association rules on the frequent itemsets at each level
def ruleGen(freqItemList,confidence):
    ruleGen = [[] for _ in range(len(freqItemList))]
    ruleGen1 = [[] for _ in range(len(freqItemList))]
    searchSpaceItemSets,searchSpaceItemSetsCount = itemSetFreq(freqItemList)
    #print(len(searchSpaceItemSets))
    #print(len(searchSpaceItemSetsCount))
    for level in range(len(freqItemList)):
        print("The Number of rules when itemset length is :",level+1)
        for itemSet in freqItemList[level]:
            combinationsGenerated = combinationsGen(itemSet[0])
            #print(combinationsGenerated)
            for combination in combinationsGenerated:
                index=searchSpaceItemSets.index(combination)
                combinationCount = searchSpaceItemSetsCount[index]
                implicationSet = list(set(itemSet[0])-set(combination))
                ruleConfidence = itemSet[1]/combinationCount
                if ruleConfidence > confidence:
                    ruleGen[level].append([combination,implicationSet,ruleConfidence])
                #if a rule is not confident then all its subsets are also not confident
                #(confidence rule obeys antimonotone property on the same itemset)
                else:
                    pruneCombinations = combinationsGen(combination)
                    #print(len(combinationsGenerated))
                    for pruneCombination in pruneCombinations:
                        if pruneCombination in combinationsGenerated:
                            combinationsGenerated.remove(pruneCombination)
        print(len(ruleGen[level]))
    return ruleGen


def start():
    dataSetSelection = int(input("Enter\n"
                                    "  1 for nursery data set(12960 instances)\n"
                                    "  2 for car data set(1729 instances)\n"
                                    "  3 for mushroom data set(8124 instances)\n"))
    if dataSetSelection == 1:
        fileName = "nursery.data.txt"
        dataPath = "nursery data set/"
    elif dataSetSelection == 2:
        fileName = "car.data.txt"
        dataPath = "car data set/"
    elif dataSetSelection == 3:
        fileName = "agaricus-lepiota.data.txt"
        dataPath = "mushroom data set/"
    elif dataSetSelection == 4:
        fileName = "sample.txt"
        dataPath = "sample data set"
    print("Loading Please wait.........")
    preprocessor(dataPath,fileName)
    transactionDetails = transaction_list_gen(dataPath,fileName)
    write_data_pickle(dataPath,transactionDetails,"matrixCreation")
    transactionDetails = read_data_pickle(dataPath,"matrixCreation")
    method = int(input("Enter 1 for F(k-1)*f(k) and 2 for F(k-1)*f(k-1):\n"))
    minSup = int(input("Enter the minimum support threshold:\n"))
    confidence = int(input("Enter the minimum confidence value(make sure it is a percentage between 0-100 integer value):\n"))/100
    #startTime = time.time()
    candidateItemList,freqItemList = item_freq_counter(transactionDetails,method,minSup)
    #endTime = time.time()
    #print("The time taken to complete the process is %f"%(endTime-startTime),"sec")
    rulesGenerated = ruleGen(freqItemList,confidence)



start()