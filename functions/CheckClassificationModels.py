
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
import random
from datetime import datetime


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
    print('Total iterations will be perform (for each featureSet) = {}'.format(len(list(ParameterGrid(paramGrid)))))

    for iterCheck in list(ParameterGrid(paramGrid)):
        startIterTime = time.time()
        result = check_clf_models(iterCheck['dataMods'][1], dataTarget, RS, iterCheck['cv_strategy'], n_jobs, debugMode)
        results.append([iterCheck['dataMods'][0], iterCheck['cv_strategy'], result])
        print('Iteration done in {} [mins]'.format(round((time.time() - startIterTime) / 60., 2)))

    print('Congrats! All DONE. Total time is [mins]: {}\n'.format(round((time.time() - startGlobalTime) / 60., 2)))
    return results

#use this method for debug while adding new model to list
def check_clf_model_candidate(data, dataTarget, RS, cv, n_jobs, debugMode):

    result = pd.DataFrame(columns=['Model', 'Acc.', 'Std'])

    ### enter candidate code here

    result.sort_values(by='Acc.', ascending=False, inplace=True)
    result.loc[len(result)] = ['TOTAL AVG', result['Accuracy'].mean(), result['Std'].mean()]
    return result


def check_clf_models(data, dataTarget, RS, cv, n_jobs, debugMode):

    result = pd.DataFrame(columns=['Model', 'Acc.', 'Std'])

    if debugMode:
        print("SGDClassifier calculating ... ")

    SGDModel = SGDClassifier(random_state=RS)
    scores = cross_val_score(SGDModel, data, dataTarget, cv=cv, n_jobs=n_jobs)
    result.loc[len(result)] = ['SGDClassifier model', scores.mean(), scores.std()]

    if debugMode:
        print("SGDClassifier DONE. Accuracy: {}, std: {}".format(scores.mean(), scores.std()))
        print("KNeighborsClassifier calculating ... ")


    Kngbh = KNeighborsClassifier(n_neighbors=3)
    scores = cross_val_score(Kngbh, data, dataTarget, cv=cv, n_jobs=n_jobs)
    result.loc[len(result)] = ['KNeighborsClassifier ', scores.mean(), scores.std()]

    if debugMode:
        print("KNeighborsClassifier DONE. Accuracy: {}, std: {}".format(scores.mean(), scores.std()))
        print("SVClinear calculating ... ")

    SVClinear = LinearSVC()
    scores = cross_val_score(SVClinear, data, dataTarget, cv=cv, n_jobs=n_jobs)
    result.loc[len(result)] = ['SVClinear', scores.mean(), scores.std()]

    if debugMode:
        print("SVClinear DONE. Accuracy: {}, std: {}".format(scores.mean(), scores.std()))
        print("SVC calculating ... ")

    SVCclf = SVC()
    scores = cross_val_score(SVCclf, data, dataTarget, cv=cv, n_jobs=n_jobs)
    result.loc[len(result)] = ['SVC', scores.mean(), scores.std()]

    if debugMode:
        print("SVC DONE. Accuracy: {}, std: {}".format(scores.mean(), scores.std()))
        print("GaussianNB calculating ... ")

    gausModel = GaussianNB()
    scores = cross_val_score(gausModel, data, dataTarget, cv=cv, n_jobs=n_jobs)
    result.loc[len(result)] = ['GaussianNB ', scores.mean(), scores.std()]

    if debugMode:
        print("GaussianNB DONE. Accuracy: {}, std: {}".format(scores.mean(), scores.std()))
        print("LinearRegression calculating ... ")

    lrModel = LinearRegression()
    scores = cross_val_score(lrModel, data, dataTarget, cv=cv, n_jobs=n_jobs, scoring=linear_scorer)
    result.loc[len(result)] = ['LinearRegression', scores.mean(), scores.std()]

    if debugMode:
        print("LinearRegression DONE. Accuracy: {}, std: {}".format(scores.mean(), scores.std()))
        print("LogisticRegression calculating ... ")

    lgrModel = LogisticRegression(random_state=RS)
    scores = cross_val_score(lgrModel, data, dataTarget, cv=cv, n_jobs=n_jobs, scoring=linear_scorer)
    result.loc[len(result)] = ['LogisticRegression ', scores.mean(), scores.std()]

    if debugMode:
        print("LogisticRegression DONE. Accuracy: {}, std: {}".format(scores.mean(), scores.std()))
        print("RandomForestClassifier calculating ... ")

    rForest = RandomForestClassifier(random_state=RS)
    scores = cross_val_score(rForest, data, dataTarget, cv=cv, n_jobs=n_jobs)
    result.loc[len(result)] = ['RandomForestClassifier (auto)', scores.mean(), scores.std()]

    if debugMode:
        print("RandomForestClassifier DONE. Accuracy: {}, std: {}".format(scores.mean(), scores.std()))
        print("XGBClassifier calculating ... ")

    XGBModel = xgb.XGBClassifier(random_state=RS, n_jobs=n_jobs)
    scores = cross_val_score(XGBModel, data, dataTarget, cv=cv, n_jobs=n_jobs)
    result.loc[len(result)] = ['XGBClassifier ', scores.mean(), scores.std()]

    if debugMode:
        print("XGBClassifier DONE. Accuracy: {}, std: {}".format(scores.mean(), scores.std()))
        print("DecisionTreeClassifier calculating ... ")

    desTree = DecisionTreeClassifier(random_state=RS)
    scores = cross_val_score(desTree, data, dataTarget, cv=cv, n_jobs=n_jobs)
    result.loc[len(result)] = ['DecisionTreeClassifier ', scores.mean(), scores.std()]

    if debugMode:
        print("DecisionTreeClassifier DONE. Accuracy: {}, std: {}".format(scores.mean(), scores.std()))
        print("Perceptron calculating ... ")

    prcModel = Perceptron()
    scores = cross_val_score(prcModel, data, dataTarget, cv=cv, n_jobs=n_jobs)
    result.loc[len(result)] = ['Perceptron ', scores.mean(), scores.std()]

    if debugMode:
        print("Perceptron DONE. Accuracy: {}, std: {}".format(scores.mean(), scores.std()))
        print("LightGBM calculating ... ")

    lgbModel = lgb.LGBMClassifier(random_state=RS)
    scores = cross_val_score(lgbModel, data, dataTarget, cv=cv, n_jobs=n_jobs)
    result.loc[len(result)] = ['LightGBM ', scores.mean(), scores.std()]

    if debugMode:
        print("LightGBM DONE. Accuracy: {}, std: {}".format(scores.mean(), scores.std()))

    result.sort_values(by='Acc.', ascending=False, inplace=True)
    result.loc[len(result)] = ['TOTAL AVG', result['Acc.'].mean(), result['Std'].mean()]

    result['Pos.'] = [i for i in range(1, len(result) + 1)]
    result = result.set_index('Model')
    return result


def postprocess_and_save_results(results, selectedFeatures, cvs, dataTrainMods, path):
    # rename all headers
    i = 0
    for result in results:
        postfix = str(i)
        result[2].columns = [newName + postfix for newName in list(result[2].columns)]
        i += 1

    # join all results to summaryTable
    summaryTable = pd.DataFrame(results[0][2].index).set_index('Model')
    for result in results:
        summaryTable = summaryTable.join(result[2])

    # save iterations have been performed
    paramGrid = {'cv_strategy': cvs, 'ScaledOrNot': dataTrainMods}
    legend = pd.DataFrame(columns=['cv_strategy', 'ScaledOrNot'])

    for iterParams in list(ParameterGrid(paramGrid)):
        legend.loc[len(legend)] = [iterParams.get('cv_strategy'), iterParams.get('ScaledOrNot')[0]]

    # add additional agg info to summaryTable
    positionColumns = ['Pos.' + str(c) for c in range(0, len(results))]
    summaryTable['Sum Pos.'] = summaryTable[positionColumns].sum(axis=1)

    accuracyColumns = ['Acc.' + str(c) for c in range(0, len(results))]
    summaryTable['Total Acc.'] = summaryTable[accuracyColumns].sum(axis=1)

    summaryTableAgg = summaryTable[positionColumns + ['Total Acc.', 'Sum Pos.']].sort_values(by='Total Acc.',
                                                                                             ascending=False)

    # all to files for analysing
    digit = random.randrange(9999)  # to group results
    folder = path + str(digit) + str('--')

    (pd.DataFrame(selectedFeatures, columns=['featureName'])
     .to_html(
        folder + 'featureList {}.html'.format(datetime.now().strftime("%Y-%m-%d--%H-%M"))))  # save list of features

    (summaryTable.sort_values(by='Total Acc.', ascending=False)
     .to_html(folder + 'summaryTableFull {}.html'.format(datetime.now().strftime("%Y-%m-%d--%H-%M"))))
    (summaryTableAgg[positionColumns + ['Total Acc.', 'Sum Pos.']]
     .to_html(folder + 'summaryTableAgg {}.html'.format(datetime.now().strftime("%Y-%m-%d--%H-%M"))))
    legend.to_html(folder + 'summaryLegend {}.html'.format(datetime.now().strftime("%Y-%m--%d_%H-%M")))
    print('All files for group {} were succesfully saved to disk.\n'.format(digit))

    return 0