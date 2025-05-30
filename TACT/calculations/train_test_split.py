import pandas as pd
import numpy as np
import copy

def train_test_split(trainPercent, inputdata, stepOverride = False):
    '''
    train is 'split' == True
    '''

    _inputdata = pd.DataFrame(columns=inputdata.columns, data=copy.deepcopy(inputdata.values))

    if stepOverride:
        msk = [False] * len(inputdata)
        _inputdata['split'] = msk
        _inputdata.loc[stepOverride[0]:stepOverride[1], 'split'] =  True

    else:
        msk = np.random.rand(len(_inputdata)) < float(trainPercent/100)
        # train = _inputdata[msk]
        # test = _inputdata[~msk]
        _inputdata['split'] = msk

    return _inputdata