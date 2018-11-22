
import pandas as pd
#import numpy as np
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

    result = pd.DataFrame(columns=['Model', 'Acc.', 'Std', 'Time'])

    ### enter candidate code here

    result.sort_values(by='Acc.', ascending=False, inplace=True)
    result.loc[len(result)] = ['TOTAL AVG', result['Accuracy'].mean(), result['Std'].mean()]
    return result


def check_clf_models(data, dataTarget, RS, cv, n_jobs, debugMode):

    result = pd.DataFrame(columns=['Model', 'Acc.', 'Std', 'Time'])

    if debugMode:
        print("SGDClassifier calculating ... ")

    startIterTime = time.time()
    SGDModel = SGDClassifier(random_state=RS)
    scores = cross_val_score(SGDModel, data, dataTarget, cv=cv, n_jobs=n_jobs)
    result.loc[len(result)] = ['SGDClassifier model', scores.mean(), scores.std(),
                               round((time.time() - startIterTime) / 60., 2)]

    if debugMode:
        print("SGDClassifier DONE. Accuracy: {}, std: {}".format(scores.mean(), scores.std()))
        print("KNeighborsClassifier calculating ... ")

    startIterTime = time.time()
    Kngbh = KNeighborsClassifier(n_neighbors=3)
    scores = cross_val_score(Kngbh, data, dataTarget, cv=cv, n_jobs=n_jobs)
    result.loc[len(result)] = ['KNeighborsClassifier', scores.mean(), scores.std(),
                               round((time.time() - startIterTime) / 60., 2)]

    if debugMode:
        print("KNeighborsClassifier DONE. Accuracy: {}, std: {}".format(scores.mean(), scores.std()))
        print("SVClinear calculating ... ")

    startIterTime = time.time()
    SVClinear = LinearSVC()
    scores = cross_val_score(SVClinear, data, dataTarget, cv=cv, n_jobs=n_jobs)
    result.loc[len(result)] = ['SVClinear', scores.mean(), scores.std(), round((time.time() - startIterTime) / 60., 2)]

    if debugMode:
        print("SVClinear DONE. Accuracy: {}, std: {}".format(scores.mean(), scores.std()))
        print("SVC calculating ... ")

    startIterTime = time.time()
    SVCclf = SVC()
    scores = cross_val_score(SVCclf, data, dataTarget, cv=cv, n_jobs=n_jobs)
    result.loc[len(result)] = ['SVC', scores.mean(), scores.std(), round((time.time() - startIterTime) / 60., 2)]

    if debugMode:
        print("SVC DONE. Accuracy: {}, std: {}".format(scores.mean(), scores.std()))
        print("GaussianNB calculating ... ")

    startIterTime = time.time()
    gausModel = GaussianNB()
    scores = cross_val_score(gausModel, data, dataTarget, cv=cv, n_jobs=n_jobs)
    result.loc[len(result)] = ['GaussianNB ', scores.mean(), scores.std(), round((time.time() - startIterTime) / 60., 2)]

    if debugMode:
        print("GaussianNB DONE. Accuracy: {}, std: {}".format(scores.mean(), scores.std()))
        print("LinearRegression calculating ... ")

    startIterTime = time.time()
    lrModel = LinearRegression()
    scores = cross_val_score(lrModel, data, dataTarget, cv=cv, n_jobs=n_jobs, scoring=linear_scorer)
    result.loc[len(result)] = ['LinearRegression', scores.mean(), scores.std(), round((time.time() - startIterTime) / 60., 2)]

    if debugMode:
        print("LinearRegression DONE. Accuracy: {}, std: {}".format(scores.mean(), scores.std()))
        print("LogisticRegression calculating ... ")

    startIterTime = time.time()
    lgrModel = LogisticRegression(random_state=RS)
    scores = cross_val_score(lgrModel, data, dataTarget, cv=cv, n_jobs=n_jobs, scoring=linear_scorer)
    result.loc[len(result)] = ['LogisticRegression ', scores.mean(), scores.std(), round((time.time() - startIterTime) / 60., 2)]

    if debugMode:
        print("LogisticRegression DONE. Accuracy: {}, std: {}".format(scores.mean(), scores.std()))
        print("RandomForestClassifier calculating ... ")

    startIterTime = time.time()
    rForest = RandomForestClassifier(random_state=RS)
    scores = cross_val_score(rForest, data, dataTarget, cv=cv, n_jobs=n_jobs)
    result.loc[len(result)] = ['RandomForestClassifier (auto)', scores.mean(), scores.std(), round((time.time() - startIterTime) / 60., 2)]

    if debugMode:
        print("RandomForestClassifier DONE. Accuracy: {}, std: {}".format(scores.mean(), scores.std()))
        print("XGBClassifier calculating ... ")

    startIterTime = time.time()
    XGBModel = xgb.XGBClassifier(random_state=RS, n_jobs=n_jobs)
    scores = cross_val_score(XGBModel, data, dataTarget, cv=cv, n_jobs=n_jobs)
    result.loc[len(result)] = ['XGBClassifier ', scores.mean(), scores.std(), round((time.time() - startIterTime) / 60., 2)]

    if debugMode:
        print("XGBClassifier DONE. Accuracy: {}, std: {}".format(scores.mean(), scores.std()))
        print("DecisionTreeClassifier calculating ... ")

    startIterTime = time.time()
    desTree = DecisionTreeClassifier(random_state=RS)
    scores = cross_val_score(desTree, data, dataTarget, cv=cv, n_jobs=n_jobs)
    result.loc[len(result)] = ['DecisionTreeClassifier ', scores.mean(), scores.std(), round((time.time() - startIterTime) / 60., 2)]

    if debugMode:
        print("DecisionTreeClassifier DONE. Accuracy: {}, std: {}".format(scores.mean(), scores.std()))
        print("Perceptron calculating ... ")

    startIterTime = time.time()
    prcModel = Perceptron()
    scores = cross_val_score(prcModel, data, dataTarget, cv=cv, n_jobs=n_jobs)
    result.loc[len(result)] = ['Perceptron ', scores.mean(), scores.std(), round((time.time() - startIterTime) / 60., 2)]

    if debugMode:
        print("Perceptron DONE. Accuracy: {}, std: {}".format(scores.mean(), scores.std()))
        print("LightGBM calculating ... ")

    startIterTime = time.time()
    lgbModel = lgb.LGBMClassifier(random_state=RS)
    scores = cross_val_score(lgbModel, data, dataTarget, cv=cv, n_jobs=n_jobs)
    result.loc[len(result)] = ['LightGBM ', scores.mean(), scores.std(), round((time.time() - startIterTime) / 60., 2)]

    if debugMode:
        print("LightGBM DONE. Accuracy: {}, std: {}".format(scores.mean(), scores.std()))

    result.sort_values(by='Acc.', ascending=False, inplace=True)
    result.loc[len(result)] = ['TOTAL AVG', result['Acc.'].mean(), result['Std'].mean(), result['Time'].mean()]

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


    # all to file
    featureList = pd.DataFrame(selectedFeatures, columns=['featureName']).T

    summaryTable = summaryTable.sort_values(by='Total Acc.', ascending=False)

    summaryTableAgg = summaryTable[positionColumns + ['Total Acc.', 'Sum Pos.']].sort_values(by='Total Acc.',
                                                                                             ascending=False)

    with open("{}{} - result ({} features).html".format(path, datetime.now().strftime("%Y-%m-%d--%H-%M"), str(len(selectedFeatures))), 'w') as _file:
        _file.write('File generated: {}'.format(datetime.now().strftime("%Y-%m-%d--%H-%M")) + '<br>')
        _file.write('Features list: <br>{}'.format(featureList.to_html()) + '<br>')
        _file.write('Summary agg table: <br>{}'.format(summaryTableAgg.to_html()) + '<br>')
        _file.write('Full summary table: <br>{}'.format(summaryTable.to_html()) + '<br>')
        _file.write('Legend of iterations table (see columns postfix of header of summary table : <br>{}'.format(legend.to_html()))

    print('All files were successfully saved to disk...')

    return 0