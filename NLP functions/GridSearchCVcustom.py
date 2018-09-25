from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ParameterGrid
import time

def get_pipeline2 (vectorizer, classifier):
    return Pipeline(
            [("vect", vectorizer),
            ("clf", classifier)]
        )

def get_pipeline3(vectorizer, transformer, classifier):
    return Pipeline(
            [("vect", vectorizer),
            ("trans", transformer),
            ("clf", classifier)]
        )

def detect_search_params(search):
    #TODO: add processing for transformer (get_pipeline3)
    clf = []
    clf_params = {}
    vect = []
    vect_params = {}

    for param in search.keys():
        if param == 'clf':
            clf = search[param]
            continue
        if param == 'vect':
            vect = search[param]
            continue
        if 'vect__' in param:
            vect_params[param.replace('vect__', '')] = search[param]
            continue
        if 'clf__' in param:
            clf_params[param.replace('clf__', '')] = search[param]
            continue
        raise Exception('Unparsed param \'{}\' was found. Review your process'.format(param))

    return clf, clf_params, vect, vect_params


def GridSearchCVcustom(pipeline, param_grid, data, labels,  cv=5, n_jobs=-1, scoring='accuracy', printNotes=True,
                       countdownElems=10, pre_dispatch=1, showSteps=False):
    startGlobalTime = time.time()

    bestScore = 0
    bestScoreParams = 0
    
    totalIterations = len(list(ParameterGrid(param_grid)))
    currentIteration = 1
    
    print('Total iterations will be performed: {}'.format(totalIterations))

    if showSteps:
        print('These steps will be performed:')

        idx = 0
        for search in (ParameterGrid(param_grid)):
            print('Iter {}:'.format(idx+1), end=' ')
            for key, value in (search.items()):
                if key != 'clf' and key != 'vect':
                    print('{}: {}'.format(key, value), end='  ')
            idx += 1
            print('\n', end='')

    for search in list(ParameterGrid(param_grid)):

        clf, clf_params, vect, vect_params = detect_search_params(search)
        pipeline.set_params(**{'vect': (type(vect))(**vect_params)}, **{'clf': (type(clf))(**clf_params)})
        try:
            startIterTime = time.time()
            scores = cross_val_score(pipeline, data, labels, cv=cv, scoring=scoring, n_jobs=n_jobs, pre_dispatch=pre_dispatch)
        except:
            print('!!!!! Something went wrong with iteration {}, search params = {}. Going next ?!?!...'.format(currentIteration, pipeline.named_steps ))
            currentIteration += 1
            # continue # for experimental!
            break
        if float(scores.mean()) > bestScore:
            bestScore = scores.mean()
            bestScoreParams = pipeline.named_steps 
            if printNotes:
                print('New best result = {} (std = {}) during {} [secs] (iter = {}) for search with params \n{}\n'. format(round(bestScore, 5), round(scores.std(),5), round(time.time() - startIterTime, 0), currentIteration, search.items()))

        currentIteration += 1

        if printNotes and currentIteration > 0 and ((totalIterations - currentIteration) % countdownElems == 0):
            print('Iterations to perform: {}'.format(totalIterations - currentIteration))
            avgIterTime = (time.time() - startGlobalTime)/currentIteration * 1.
            print('Average time for iteration [secs]: {}'.format(round(avgIterTime, 1)))
            print('Job will be done in (approximately) [mins] : {}\n'.format(round(avgIterTime * (totalIterations - currentIteration) / 60., 1)))

    print('Average time for iteration [secs]: {}'.format(round((time.time() - startGlobalTime) / 60., 1)))
    print('Congrats! All DONE. Total time is [mins]: {}'.format(round((time.time() - startGlobalTime) / 60., 2)))
    return bestScore, bestScoreParams