from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ParameterGrid
import time
import traceback

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
                       countdownElems=5, pre_dispatch=1, showSteps=False, itersToPerform = None):

    if itersToPerform == 'All':
        itersToPerform = [i+1 for i in range(0, len(list(ParameterGrid(param_grid))))]

    print('Total iterations exist for provided gridSearch params: {}'.format(len(list(ParameterGrid(param_grid)))))

    if showSteps:
        print('These steps available to be performed:')

        idx = 1
        for search in (ParameterGrid(param_grid)):
            if itersToPerform is None:
                print('Iter {}:'.format(idx), end=' ')
            else:
                print('{}Iter {}:'.format('v ' if idx in itersToPerform else '  ', idx), end=' ')

            for key, value in (search.items()):
                if key != 'clf' and key != 'vect':
                    print('{}: {}'.format(key, value), end='  ')
            idx += 1
            print('\n', end='')

    if itersToPerform is None:
        print('No iterations will be really performed, exiting')
        return 0, 0, 0

    startGlobalTime = time.time()
    bestScore = 0
    bestScoreParams = 0
    bestIter = 1

    totalIterations = len(set(itersToPerform) & set(range(1, len(list(ParameterGrid(param_grid))))))
    currentIteration = itersToPerform[0]
    currentIterationNum = 0

    print('\n')
    print('--==Calculations...==--')
    for search in list(ParameterGrid(param_grid)):

        if currentIteration in itersToPerform:

            currentIterationNum += 1
            clf, clf_params, vect, vect_params = detect_search_params(search)
            pipeline.set_params(**{'vect': (type(vect))(**vect_params)}, **{'clf': (type(clf))(**clf_params)})
            try:
                startIterTime = time.time()
                scores = cross_val_score(pipeline, data, labels, cv=cv, scoring=scoring, n_jobs=n_jobs,
                                         pre_dispatch=pre_dispatch)
            except:
                print(traceback.format_exc())
                print('!!!!! Something went wrong with iteration {}, search params = {}. Going next ?!?!...'.format(
                    currentIteration, pipeline.named_steps))
                currentIteration += 1
                continue # for experimental!
                # break
            if float(scores.mean()) > bestScore:
                bestScore = scores.mean()
                bestScoreParams = pipeline.named_steps
                bestIter = currentIteration
                if printNotes:
                    print(
                        'New best result = {} (std = {}) during {} [secs] (iter = {}) for search with params \n{}\n'.format(
                            round(bestScore, 5), round(scores.std(), 5), round(time.time() - startIterTime, 0),
                            currentIteration, search.items()))

            if printNotes and currentIterationNum > 0 and ((totalIterations - currentIterationNum) % countdownElems == 0):
                print('Iterations to perform: {}'.format(totalIterations - currentIterationNum))
                avgIterTime = (time.time() - startGlobalTime)/currentIterationNum * 1.
                print('Average time for iteration [mins]: {}'.format(round(avgIterTime / 60., 2)))
                print('Job will be done in (approximately) [mins] : {}\n'.format(round(avgIterTime * (totalIterations - currentIterationNum) / 60., 1)))
                      #todo: may be added destination time
                      #time.strftime('%H:%M:%S', time.localtime(avgIterTime * (totalIterations - currentIterationNum) - startGlobalTime))))

        currentIteration += 1

    print('Congrats! All DONE. Total time is [mins]: {}'.format(round((time.time() - startGlobalTime) / 60., 2)))
    print('BestIter was {}, bestScore = {}, bestScoreParams = {}'.format(bestIter, bestScore, bestScoreParams))
    return bestScore, bestScoreParams, bestIter
