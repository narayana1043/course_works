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

def read_data_pickle(data_path, name):
    # reading pickled data from disk
    with open(data_path + name + "_pickle", "rb") as foo:
        unpickled_data = pickle.load(foo)
    return unpickled_data

def write_data_pickle(data_path, data_to_pickle, name):
    # pickling data to disk
    with open(data_path + name + "_pickle", "wb") as foo:
        pickle.dump(data_to_pickle, foo)

def transaction_list_gen(data_path, file_name):
    # transaction list(pandas dataframe) generation as Zeros and Ones as
    #  mentioned in the question
    with open(data_path + file_name + "PrepocessedData.txt") as transactions:
        # with open(data_path+file_name) as transactions:
        transactions_details = []
        items_list = []
        for row in transactions.read().splitlines():
            transaction = row.split(',')
            items_list += transaction
            transactions_details.append(sorted(transaction))
        items_set = sorted(set(items_list))
        item_set_len = len(items_set)
        df = pd.DataFrame(columns=items_set)
        for i, transaction in zip(range(len(transactions_details)),
                                  transactions_details):
            transaction_numpy_array = np.zeros(item_set_len)
            for item in transaction:
                transaction_numpy_array[items_set.index(item)] = 1
            df.loc[i] = transaction_numpy_array
    return df


# #function to pickle the data into the disk
# write_data_pickle(transaction_list_gen("groceries.csv"),"matrixCreation")
# #function to read pickled data from disk
# transaction_details = read_data_pickle("matrixCreation")
# print(transaction_details.head())


def freq_item_set_gen(item_sets_freq, min_sup, transaction_details):
    # Function to create frequent Item sets
    freq_dict = []
    for item_listFreq in item_sets_freq:
        frequency = transaction_details.reindex(columns=item_listFreq[0]).prod(
            axis=1).sum()
        if frequency >= int(min_sup):
            freq_dict.append([item_listFreq[0], frequency])
    # orderedfreq_dict = sorted(freq_dict, key=operator.itemgetter(1),
    #                           reverse=True)
    # return orderedfreq_dict
    return freq_dict

def multi_item_list_gen(items_list_freq):
    # Function to create multi item sets(This function implements
    #  f(k-1) * f(k-1) to generate f(k) multi item list)
    freqitems_list = [items_list[0] for items_list in items_list_freq]
    extendeditems_list = []
    if freqitems_list != []:
        if len(freqitems_list[0]) >= 2:
            for items in freqitems_list:
                for i in range(freqitems_list.index(items) + 1,
                               len(freqitems_list)):
                    if items[:-1] == freqitems_list[i][:-1]:
                        # comparing various f(k-1)*f(k-1) to generate fk term
                        extendeditems_list.append(
                            [sorted(items + [freqitems_list[i][-1]]), 0])
        elif len(freqitems_list[0]) == 1:
            extendeditems_list = multi_item_list_gen1(items_list_freq,
                                                      items_list_freq)
    return extendeditems_list

def multi_item_list_gen1(items_list_freq, freq1_item_list):
    # Function to create multi item sets(This function implements
    #  f(k-1) * f(k1) to generate f(k) multi item list)
    freq_item_list = sorted(
        [items_list_freq[0] for items_list_freq in items_list_freq])
    extendeditems_list = []
    for item_list in freq_item_list:
        # loops only if there is an F(k-1) list exists
        for item in sorted(freq1_item_list):
            items_combination = sorted(item_list + item[0])
            # print(item_list)
            # print(item[0][0])
            # print(item[0][0] not in item_list)
            if item_list[-1] < item[0][0] and [items_combination, 0] not in \
                    extendeditems_list:
                # using lexicographical order to prune reoccuring itemsets
                # print(sorted(item_list+item[0]))
                extendeditems_list.append([items_combination, 0])
    return extendeditems_list


# Output generation-- Prints a the number of itemsets along with a small sample
def printer(candidate_item_list, freq_item_list, item_list_len):
    # if os.path.isfile(data_path+"assRules1Output.txt") == True:
    #     os.remove(data_path+"assRules1Output.txt")
    # sys.stdout = open(data_path+"assRules1Output.txt","w")
    print("*********Number of items when the list length is ",
          item_list_len + 1, "*********")
    # print("Candidate item lists: ", candidate_item_list[0:2], "........",
    #       candidate_item_list[-2:])
    print("Count of Candidate item lists: ", len(candidate_item_list))
    # print("Frequent item lists: ", freq_item_list[0:2], "........",
    #       freq_item_list[-2:])
    print("Count of Frequent item lists: ", len(freq_item_list))

def printer1(candidate_item_list, freq_item_list, item_list_len):
    # Output generation-- Prints a only the number of itemsets
    print("*********Number of items when the list length is ",
          item_list_len + 1, "*********")
    print("Count of Candidate item lists: ", len(candidate_item_list))

    print("Count of Frequent item lists: ", len(freq_item_list))

def item_freq_counter(transaction_details, method, min_sup):
    # Function to make a list of columns that are passed to the
    #  freq_item_set_gen function to get the freq of each item
    candidate_item_list = []
    freq_item_list = []
    item_list_len = 0
    item_list = [[[transaction_details.columns.values[i]], 0] for i in
                 range(len(transaction_details.columns.values))]
    candidate_item_list.append(item_list)
    freq_item_list.append(item_list)
    candidate_item_list[item_list_len] = freq_item_set_gen(
        candidate_item_list[item_list_len], 0,
        transaction_details)  # min_sup = 0(frequent item generation)
    freq_item_list[item_list_len] = \
        freq_item_set_gen(freq_item_list[item_list_len], min_sup,
                          transaction_details)
    # min_sup = user input value
    printer1(candidate_item_list[item_list_len], freq_item_list[
        item_list_len], 0)
    # for item_list_len in range(1,item_set_len):
    while True:
        item_list_len += 1
        if method == 1:
            item_list = \
                multi_item_list_gen1(freq_item_list[item_list_len - 1],
                                     candidate_item_list[0])
        elif method == 2:
            item_list = multi_item_list_gen(freq_item_list[item_list_len - 1])
        candidate_item_list.append(item_list)
        freq_item_list.append(item_list)
        candidate_item_list[item_list_len] = freq_item_set_gen(
            candidate_item_list[item_list_len], 0,
            transaction_details)  # min_sup = 0(frequent item generation)
        freq_item_list[item_list_len] = freq_item_set_gen(
            freq_item_list[item_list_len], min_sup,
            transaction_details)  # min_sup = user input value
        printer1(candidate_item_list[item_list_len], freq_item_list[
            item_list_len], item_list_len)
        if len(candidate_item_list[item_list_len]) == 0:
            break
    return candidate_item_list, freq_item_list


# function to generate the preprocessed file
def preprocessor(data_path, file_name):
    df = pd.DataFrame.from_csv(data_path + file_name, index_col=False)
    for column in df.columns.values:
        df[column] = column + df[column]
    # print(df)
    df.to_csv(data_path + file_name + "PrepocessedData.txt", index=False,
              header=False)


def start():
    data_set_selection = int(input("Enter\n "
                                   "1 for nursery data set(12960 instances)\n"
                                   "2 for car data set(1729 instances)\n"
                                   "3 for mushroom data set(8124 instances)\n")
                             )
    if data_set_selection == 1:
        file_name = "nursery.data.txt"
        data_path = "nursery data set/"
    elif data_set_selection == 2:
        file_name = "car.data.txt"
        data_path = "car data set/"
    elif data_set_selection == 3:
        file_name = "agaricus-lepiota.data.txt"
        data_path = "mushroom data set/"
    elif data_set_selection == 4:
        file_name = "sample.txt"
        data_path = "sample data set"
    print("Loading Please wait.........")
    preprocessor(data_path, file_name)
    transaction_details = transaction_list_gen(data_path, file_name)
    write_data_pickle(data_path, transaction_details, "matrixCreation")
    transaction_details = read_data_pickle(data_path, "matrixCreation")
    method = int(input("Enter 1 for F(k-1)*f(k) and 2 for F(k-1)*f(k-1):\n"))
    min_sup = int(input("Enter the minimum support threshold:\n"))
    start_time = time.time()
    candidate_item_list, freq_item_list = \
        item_freq_counter(transaction_details, method, min_sup)
    end_time = time.time()
    print("The time taken to complete the process is %f" % (end_time -
                                                            start_time), "sec")


start()
