import pandas as pd
#import numpy as np
from sklearn.externals import joblib
from sklearn import preprocessing

def fe_cat_ohe(dataAll, col, prefix='OHE', params=None):

    # generate OHE using pd
    data_ = pd.get_dummies(dataAll[col], prefix='{}_{}'.format(col, prefix))
    return data_, list(data_.columns)


def save_to_file(feature, featureName, path):
    _ = joblib.dump(feature, open('{}{}.pkl'.format(path, featureName), 'wb'), 9)
    print('{} saved.'.format(featureName))
    return 0


def fe_cat_le(dataAll, col, prefix='LE', params=None):
    le = preprocessing.LabelEncoder()
    le.fit(dataAll[col])
    data_ = pd.DataFrame(le.transform(dataAll[col]), columns=(['{}_{}'.format(col, prefix)]))

    return data_, list(data_.columns)


def create_dataset_from_features(featureList, path, params=None):
    dataset = None
    for feature in featureList:
        if dataset is None:
            dataset = joblib.load('{}{}.pkl'.format(path, feature))
        else:
            dataset = pd.concat([dataset, joblib.load('{}{}.pkl'.format(path, feature))], axis=1)

    return dataset