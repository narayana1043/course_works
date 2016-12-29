import numpy as np
import pandas as pd

def read_data():
    '''
    reads data into a data frame csv into python pandas dataframe
    :return: train and test Dataframes
    '''
    train = pd.DataFrame.from_csv(path='./data/train.csv')
    test = pd.DataFrame.from_csv(path='./data/test.csv')
    return train, test
