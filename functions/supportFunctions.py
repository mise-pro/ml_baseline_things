import pandas as pd
import numpy as np


def load_raw_data(folderPath='data_raw/'):
    dataTrain = pd.read_csv(folderPath + 'train.csv')
    dataTest = pd.read_csv(folderPath + 'test.csv')
    dataAll = pd.concat([dataTrain, dataTest])
    print('Train {}; Test {}; Total {}.'.format(dataTrain.shape, dataTest.shape, dataAll.shape))
    return dataTrain, dataTest, dataAll


def check_missing_data(dataTrain, dataTest, featuresList=[]):
    if len(featuresList) == 0:
        print('Train:\n', dataTrain.isnull().sum(), '\n')
        print('Test:\n', dataTest.isnull().sum())
    else:
        featuresList = list(set(dataTrain.columns.values) & set(dataTest.columns.values) & set(featuresList))
        print('Train:\n', dataTrain[featuresList].isnull().sum(), '\n')
        print('Test:\n', dataTest[featuresList].isnull().sum())
    return 0

def check_unique_data(dataTrain, dataTest, column):
    print('Train unique values:\n', dataTrain[column].unique(), '\n')
    print('Test unique values:\n', dataTest[column].unique())
    return 0