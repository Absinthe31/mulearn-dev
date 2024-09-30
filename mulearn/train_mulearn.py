import numpy as np
import sklearn
import matplotlib.pyplot as plt
import random
import fknn
from fknn import FuzzyKNN
from fcmeans import FCM
from sklearn.svm import SVC, SVR
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid
from joblib import Parallel, delayed
import logging
import json
from logging import FileHandler
import cloudpickle
import sys
import tqdm
from PIL import Image
import hashlib
import time
import os
from itertools import zip_longest

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd

from importlib import reload

import __init__
import distributions
import kernel
import fuzzifier
import optimization

reload(__init__)
reload(distributions)
reload(kernel)
reload(fuzzifier)
reload(optimization)

from __init__ import *
from distributions import *
from kernel import *
from fuzzifier import *
from optimization import *


#general testing functions for synthetic datasets
def gaussian(x, var):
    y = (np.e ** (-(((x-0.5)*2)**2)/(2*var)))
    return y

trapezoidal = lambda x: (2.3*x if x < 0.5 else -2.3*x+2.3) if (2.3*x if x < 0.5 else -2.3*x+2.3) < 1 else 1

X_test = np.linspace(0, 1, 1000)
X_test_g = np.array([x for x in X_test if (gaussian(x, np.var(X_test)) <= 0.1 or gaussian(x, np.var(X_test)) >= 0.9)])
X_test_t = np.array([x for x in X_test if (trapezoidal(x) <= 0.1 or trapezoidal(x) >= 0.9)])
X_test2_g = [np.array([x for x in X_test_g if x<0.2]),np.array([x for x in X_test_g if (x>=0.2 and x<0.8)]),np.array([x for x in X_test_g if x>0.8])]
X_test2_t = [np.array([x for x in X_test_t if x<0.2]),np.array([x for x in X_test_t if (x>=0.2 and x<0.8)]),np.array([x for x in X_test_t if x>0.8])]
X_test2 = [np.array([x for x in X_test if x<0.2]),np.array([x for x in X_test if (x>=0.2 and x<0.8)]),np.array([x for x in X_test if x>0.8])]
step = (X_test2[0][-1]-X_test2[0][0])/(len(X_test2[0])-1)

class JsonFormatter(logging.Formatter):

    @staticmethod
    def find(s, ch):
        return [i for i, ltr in enumerate(s) if ltr == ch]

    def format(self, record) -> str:
        
        log_record = {
            'timestamp': self.formatTime(record),
            'model': record.getMessage()
        }
        
        if record.exc_info:
            log_record['exception'] = self.formatException(record.exc_info)

        output = json.dumps(log_record) + ','

        output = output[:self.find(output, '"')[-1]] + '' + output[self.find(output, '"')[-1]+1:]
        output = output[:self.find(output, '"')[-1]] + '' + output[self.find(output, '"')[-1]+1:]

        output = output.replace("'",'"')
        
        return output

json_handler = FileHandler('test_logs.log')
json_handler.setLevel(logging.INFO)
json_formatter = JsonFormatter()
json_handler.setFormatter(json_formatter)

logger = logging.getLogger(__name__)
logger.addHandler(json_handler)
logging.getLogger().setLevel(logging.INFO)

def log_model(model=None, score=-1, acc=-1, std=-1, model_name='', serialize=False, dataset=''):

    if not isinstance(model, list):
        json_dict = { 'model_name' : model.__repr__() if (model_name == '') else model_name }
    else:
        json_dict = { 'model_name' : 'Unidentified model' if (model_name == '') else model_name }
    
    if score != -1:
        json_dict['score'] = score
    if acc != -1:
        json_dict['acc'] = acc
    if std != -1:
        json_dict['std'] = std
        
    if serialize == True:
        try:
    
            dir_name = hashlib.md5(int(time.time()*1000).to_bytes(8, 'big', signed=True)).hexdigest()[:10]
            os.mkdir(f'./objects/{dir_name}')
            
            if isinstance(model, list):
    
                for i,m in enumerate(model):
    
                    filehandler = open(f"objects/{dir_name}/{i+1}.obj","wb")
                    cloudpickle.dump(m, filehandler)
                    filehandler.close()
                    json_dict['obj'] = f'objects/{dir_name}/1.obj'
                
            else:
                filehandler = open(f"objects/{dir_name}/1.obj","wb")
                cloudpickle.dump(model, filehandler)
                filehandler.close()
                json_dict['obj'] = f'objects/{dir_name}/1.obj'
        except Exception as e: 
            #print("Error with object serialization")
            print(e)
            
    if dataset != '':
        json_dict['dataset'] = dataset

    return json_dict

digit = 0

path = f'datasets/MNIST/digit-{digit}/'
dataset = pd.read_json(f'datasets/MNIST/digit-{digit}/mnist_full.json')

X_train = np.array(dataset.X_train.dropna().to_list())
y_train = np.array(dataset.y_train.dropna().to_list())
X_test = np.array(dataset.X_test.to_list())
y_test = np.array(dataset.y_test.to_list())

files = [(path + '/LinearKernels/' + name) for name in os.listdir(path + '/LinearKernels/')]
linear_kernels = []
for file in files:
    with open(file, "rb") as f:
        linear_kernels.append(cloudpickle.load(f))
    

parameters_linear = {'c' : [x for x in np.logspace(-2,2,5)],
                     'k' : linear_kernels,
                     'fuzzifier' : [LinearFuzzifier(profile='infer'), LinearFuzzifier(profile='fixed')]}


files = [(path + '/PolynomialKernels/' + name) for name in os.listdir(path + '/PolynomialKernels/')]
polynomial_kernels = []
for file in files:
    with open(file, "rb") as f:
        polynomial_kernels.append(cloudpickle.load(f))

parameters_polynomial = {'c' : [x for x in np.logspace(-2,2,5)],
                         'k' : polynomial_kernels,
                         'fuzzifier' : [LinearFuzzifier(profile='infer'), LinearFuzzifier(profile='fixed')]}


files = [(path + '/GaussianKernels/' + name) for name in os.listdir(path + '/GaussianKernels/')]
gaussian_kernels = []
for file in files:
    with open(file, "rb") as f:
        gaussian_kernels.append(cloudpickle.load(f))

parameters_gaussian = {'c' : [x for x in np.logspace(-2,2,5)],
                       'k' : gaussian_kernels,
                       'fuzzifier' : [ExponentialFuzzifier(profile='fixed'), ExponentialFuzzifier(profile='infer')] +
                                     [ExponentialFuzzifier(profile='alpha',alpha=x) for x in np.logspace(-2,0,3)]}

params = [parameters_linear, parameters_polynomial, parameters_gaussian]

model = FuzzyInductor(solver=GurobiSolver(adjustment='auto'))

outer_cv_scores = []
outer_cv_params = []
inner_cv = StratifiedKFold(n_splits=5)
outer_cv = StratifiedKFold(n_splits=5)

best_models = []

for train, test in tqdm.tqdm(outer_cv.split(X_train, y_train)):

    clf = GridSearchCV(estimator=model, param_grid=params, 
                       cv=inner_cv, n_jobs=-1, verbose=0, refit=True)
    
    clf.fit(train.reshape(-1,1), y_train[train])

    m = clf.best_estimator_
    
    best_acc = 0
    for alpha in np.linspace(0.1,0.9,9):
        acc = accuracy_score(y_pred=m.predict(train.reshape(-1,1), alpha=alpha), y_true=y_train[train])
        if acc > best_acc:
            best_alpha = alpha

    outer_cv_params.append(m.__repr__())
    acc_score = accuracy_score(y_pred=m.predict(test.reshape(-1,1), alpha=best_alpha), y_true=y_train[test])
    outer_cv_scores.append(acc_score)

    best_models.append((m, acc_score, best_alpha))
    

score = [np.mean(outer_cv_scores), np.std(outer_cv_scores)]
best_models = sorted(best_models, key=lambda x:x[1], reverse=True)

print(score)
print(outer_cv_scores)

logger.info(log_model(model=best_models, model_name='mulearn', acc=score[0], std=score[1], dataset=f'MNIST_FULL_{digit}', serialize=True))


