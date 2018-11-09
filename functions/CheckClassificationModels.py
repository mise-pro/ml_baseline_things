
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
# from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import lightgbm as lgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Perceptron
from sklearn import metrics
from sklearn.model_selection import ParameterGrid
import time


def linear_scorer(estimator, x, y):
    scorer_predictions = estimator.predict(x)
    scorer_predictions[scorer_predictions > 0.5] = 1
    scorer_predictions[scorer_predictions <= 0.5] = 0
    return metrics.accuracy_score(y, scorer_predictions)

def global_check_clf_models (dataMods, dataTarget, cvs, RS, n_jobs = -1, debugMode=0):
    startGlobalTime = time.time()
    results = []

    paramGrid = {'cv_strategy': cvs,
                 'dataMods': dataMods}
    print('Total iterations will be perform = {}'.format(len(list(ParameterGrid(paramGrid)))))

    for iterCheck in list(ParameterGrid(paramGrid)):
        result = check_clf_models(iterCheck['dataMods'][1], dataTarget, RS, iterCheck['cv_strategy'], n_jobs, debugMode)
        results.append([iterCheck['dataMods'][0], iterCheck['cv_strategy'], result])
        print('Iteration done')

    print('Congrats! All DONE. Total time is [mins]: {}\n'.format(round((time.time() - startGlobalTime) / 60., 2)))
    return results

#use this method for debug while adding new model to list
def check_clf_model_candidate(data, dataTarget, RS, cv, n_jobs, debugMode):

    result = pd.DataFrame(columns=['Model', 'Accuracy', 'Std', 'Time[secs]'])

    ### enter candidate code here

    result.sort_values(by='Accuracy', ascending=False, inplace=True)
    result.loc[len(result)] = ['TOTAL AVG', result['Accuracy'].mean(), result['Std'].mean(),
                                 result['Time[secs]'].mean()]
    return result


def check_clf_models(data, dataTarget, RS, cv, n_jobs, debugMode):

    result = pd.DataFrame(columns=['Model', 'Accuracy', 'Std', 'Time[secs]'])

    if debugMode:
        print("SGDClassifier calculating ... ")

    SGDModel = SGDClassifier(random_state=RS)
    startIterTime = time.time()
    scores = cross_val_score(SGDModel, data, dataTarget, cv=cv, n_jobs=n_jobs)
    result.loc[len(result)] = ['SGDClassifier model', scores.mean(), scores.std(), round(time.time() - startIterTime, 0)]

    if debugMode:
        print("SGDClassifier DONE. Accuracy: {}, std: {}".format(scores.mean(), scores.std()))
        print("KNeighborsClassifier calculating ... ")


    Kngbh = KNeighborsClassifier(n_neighbors=3)
    startIterTime = time.time()
    scores = cross_val_score(Kngbh, data, dataTarget, cv=cv, n_jobs=n_jobs)
    result.loc[len(result)] = ['KNeighborsClassifier ', scores.mean(), scores.std(), round(time.time() - startIterTime, 0)]

    if debugMode:
        print("KNeighborsClassifier DONE. Accuracy: {}, std: {}".format(scores.mean(), scores.std()))
        print("SVClinear calculating ... ")

    SVClinear = LinearSVC()
    startIterTime = time.time()
    scores = cross_val_score(SVClinear, data, dataTarget, cv=cv, n_jobs=n_jobs)
    result.loc[len(result)] = ['SVClinear', scores.mean(), scores.std(), round(time.time() - startIterTime, 0)]

    if debugMode:
        print("SVClinear DONE. Accuracy: {}, std: {}".format(scores.mean(), scores.std()))
        print("SVC calculating ... ")

    SVCclf = SVC()
    startIterTime = time.time()
    scores = cross_val_score(SVCclf, data, dataTarget, cv=cv, n_jobs=n_jobs)
    result.loc[len(result)] = ['SVC', scores.mean(), scores.std(), round(time.time() - startIterTime, 0)]

    if debugMode:
        print("SVC DONE. Accuracy: {}, std: {}".format(scores.mean(), scores.std()))
        print("GaussianNB calculating ... ")

    gausModel = GaussianNB()
    startIterTime = time.time()
    scores = cross_val_score(gausModel, data, dataTarget, cv=cv, n_jobs=n_jobs)
    result.loc[len(result)] = ['GaussianNB ', scores.mean(), scores.std(), round(time.time() - startIterTime, 0)]

    if debugMode:
        print("GaussianNB DONE. Accuracy: {}, std: {}".format(scores.mean(), scores.std()))
        print("LinearRegression calculating ... ")

    lrModel = LinearRegression()
    startIterTime = time.time()
    scores = cross_val_score(lrModel, data, dataTarget, cv=cv, n_jobs=n_jobs, scoring=linear_scorer)
    result.loc[len(result)] = ['LinearRegression', scores.mean(), scores.std(), round(time.time() - startIterTime, 0)]

    if debugMode:
        print("LinearRegression DONE. Accuracy: {}, std: {}".format(scores.mean(), scores.std()))
        print("LogisticRegression calculating ... ")

    lgrModel = LogisticRegression(random_state=RS)
    startIterTime = time.time()
    scores = cross_val_score(lgrModel, data, dataTarget, cv=cv, n_jobs=n_jobs, scoring=linear_scorer)
    result.loc[len(result)] = ['LogisticRegression ', scores.mean(), scores.std(), round(time.time() - startIterTime, 0)]

    if debugMode:
        print("LogisticRegression DONE. Accuracy: {}, std: {}".format(scores.mean(), scores.std()))
        print("RandomForestClassifier calculating ... ")

    rForest = RandomForestClassifier(random_state=RS)
    startIterTime = time.time()
    scores = cross_val_score(rForest, data, dataTarget, cv=cv, n_jobs=n_jobs)
    result.loc[len(result)] = ['RandomForestClassifier (auto)', scores.mean(), scores.std(), round(time.time() - startIterTime, 0)]

    if debugMode:
        print("RandomForestClassifier DONE. Accuracy: {}, std: {}".format(scores.mean(), scores.std()))
        print("XGBClassifier calculating ... ")

    XGBModel = xgb.XGBClassifier(random_state=RS, n_jobs=n_jobs)
    startIterTime = time.time()
    scores = cross_val_score(XGBModel, data, dataTarget, cv=cv, n_jobs=n_jobs)
    result.loc[len(result)] = ['XGBClassifier ', scores.mean(), scores.std(),
                                 round(time.time() - startIterTime, 0)]

    if debugMode:
        print("XGBClassifier DONE. Accuracy: {}, std: {}".format(scores.mean(), scores.std()))
        print("DecisionTreeClassifier calculating ... ")

    desTree = DecisionTreeClassifier(random_state=RS)
    startIterTime = time.time()
    scores = cross_val_score(desTree, data, dataTarget, cv=cv, n_jobs=n_jobs)
    result.loc[len(result)] = ['DecisionTreeClassifier ', scores.mean(), scores.std(),
                                 round(time.time() - startIterTime, 0)]

    if debugMode:
        print("DecisionTreeClassifier DONE. Accuracy: {}, std: {}".format(scores.mean(), scores.std()))
        print("Perceptron calculating ... ")

    prcModel = Perceptron()
    startIterTime = time.time()
    scores = cross_val_score(prcModel, data, dataTarget, cv=cv, n_jobs=n_jobs)
    result.loc[len(result)] = ['Perceptron ', scores.mean(), scores.std(),
                                 round(time.time() - startIterTime, 0)]

    if debugMode:
        print("Perceptron DONE. Accuracy: {}, std: {}".format(scores.mean(), scores.std()))
        print("LightGBM calculating ... ")

    lgbModel = lgb.LGBMClassifier(random_state=RS)
    startIterTime = time.time()
    scores = cross_val_score(lgbModel, data, dataTarget, cv=cv, n_jobs=n_jobs)
    result.loc[len(result)] = ['LightGBM ', scores.mean(), scores.std(), round(time.time() - startIterTime, 0)]

    if debugMode:
        print("LightGBM DONE. Accuracy: {}, std: {}".format(scores.mean(), scores.std()))

    result.sort_values(by='Accuracy', ascending=False, inplace=True)
    result.loc[len(result)] = ['TOTAL AVG', result['Accuracy'].mean(), result['Std'].mean(),
                                 result['Time[secs]'].mean()]
    return result