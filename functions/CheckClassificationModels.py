
import pandas as pd
#import numpy as np
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
import xgboost as xgb
import lightgbm as lgb
#from sklearn.linear_model import LinearRegression
#from sklearn.linear_model import LogisticRegression
#from sklearn.linear_model import SGDClassifier
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.naive_bayes import GaussianNB

#from sklearn.tree import DecisionTreeClassifier
#from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import cross_val_score
#from sklearn.linear_model import Perceptron
from sklearn import metrics
from sklearn.model_selection import ParameterGrid
import time
from datetime import datetime


def linear_scorer(estimator, x, y):
    scorer_predictions = estimator.predict(x)
    scorer_predictions[scorer_predictions > 0.5] = 1
    scorer_predictions[scorer_predictions <= 0.5] = 0
    return metrics.accuracy_score(y, scorer_predictions)

def global_check_clf_models (dataMods, dataTarget, scoring, cvs, RS, n_jobs = -1, debugMode=0):
    startGlobalTime = time.time()
    results = []

    paramGrid = {'cv_strategy': cvs,
                 'dataMods': dataMods}
    print('Total iterations will be perform (for each featureSet) = {}'.format(len(list(ParameterGrid(paramGrid)))))

    for iterCheck in list(ParameterGrid(paramGrid)):
        startIterTime = time.time()
        result = check_clf_models(iterCheck['dataMods'][1], dataTarget, RS, scoring, iterCheck['cv_strategy'], n_jobs, debugMode)
        results.append([iterCheck['dataMods'][0], iterCheck['cv_strategy'], result])
        print('Iteration done in {} [mins]'.format(round((time.time() - startIterTime) / 60., 2)))

    print('Congrats! All DONE. Total time is [mins]: {}'.format(round((time.time() - startGlobalTime) / 60., 2)))
    return results


def check_clf_models(data, dataTarget, RS, scoring, cv, n_jobs, debugMode):

    result = pd.DataFrame(columns=['Model', 'Acc.', 'Std', 'Time'])

    models = [

        # Ensemble Methods
        [ensemble.AdaBoostClassifier(random_state=RS), 'AdaBoostClassifier (n=50)'],
        [ensemble.BaggingClassifier(random_state=RS, n_jobs=n_jobs), 'BaggingClassifier (n=10)'],
        [ensemble.ExtraTreesClassifier(random_state=RS, n_jobs=n_jobs), 'ExtraTreesClassifier (n=10)'],
        [ensemble.GradientBoostingClassifier(random_state=RS), 'GradientBoostingClassifier (n=100)'],
        [ensemble.RandomForestClassifier(random_state=RS, n_jobs=n_jobs), 'RandomForestClassifier (auto)'],

        # Gaussian Processes
        [gaussian_process.GaussianProcessClassifier(random_state=RS, n_jobs=n_jobs), 'GaussianProcessClassifier'],

        # GLM
        [linear_model.LogisticRegressionCV(cv=cv, random_state=RS, n_jobs=n_jobs), 'LogisticRegressionCV'],
        [linear_model.PassiveAggressiveClassifier(random_state=RS, n_jobs=n_jobs), 'PassiveAggressiveClassifier'],
        [linear_model.RidgeClassifierCV(cv=cv), 'RidgeClassifierCV'],
        [linear_model.SGDClassifier(random_state=RS, n_jobs=n_jobs), 'SGDClassifier'],
        [linear_model.Perceptron(random_state=RS, n_jobs=n_jobs), 'Perceptron'],

        # Navies Bayes
        [naive_bayes.BernoulliNB(), 'BernoulliNB'],
        [naive_bayes.GaussianNB(), 'GaussianNB'],

        # Nearest Neighbor
        [neighbors.KNeighborsClassifier(n_jobs=n_jobs, n_neighbors=2), 'KNeighborsClassifier (n=2)'],

        # SVM
        [svm.SVC(random_state=RS, probability=True), 'SVC'],
        [svm.NuSVC(random_state=RS, probability=True), 'NuSVC'],
        [svm.LinearSVC(random_state=RS), 'LinearSVC'],

        # Trees
        [tree.DecisionTreeClassifier(random_state=RS), 'DecisionTreeClassifier'],
        [tree.ExtraTreeClassifier(random_state=RS), 'ExtraTreeClassifier'],

        # Discriminant Analysis
        [discriminant_analysis.LinearDiscriminantAnalysis(), 'LinearDiscriminantAnalysis'],
        #[discriminant_analysis.QuadraticDiscriminantAnalysis(), 'QuadraticDiscriminantAnalysis'],   #FLOAT16 problem!

        [xgb.XGBClassifier(random_state=RS, n_jobs=n_jobs), 'XGBClassifier'],
        [lgb.LGBMClassifier(random_state=RS, n_jobs=n_jobs), 'LGBMClassifier']
    ]


    for model in models:
        if debugMode:
            print("{} calculating...".format(model[1]))

        startIterTime = time.time()

        scores = cross_val_score(model[0], data, dataTarget, scoring=scoring, cv=cv, n_jobs=n_jobs)
        result.loc[len(result)] = ['{} model'.format(model[1]), scores.mean(), scores.std(),
                                   round((time.time() - startIterTime) / 60., 2)]
        if debugMode:
            print("{} DONE. Accuracy: {}, std: {}".format(model[1], scores.mean(), scores.std()))


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

    print('All files were successfully saved to disk...\n')

    return 0