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

json_handler = FileHandler('test_logs_new.log')
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


for digit in [0]:

    print(digit)
    
    dataset = pd.read_json(f'datasets/MNIST/digit-{digit}/mnist_full.json')
    
    X_train = np.array(dataset.X_train.dropna().to_list())
    y_train = np.array(dataset.y_train.dropna().to_list())
    X_test = np.array(dataset.X_test.to_list())
    y_test = np.array(dataset.y_test.to_list())
    
    parameters = {'k' : [3,5,7,9,11]}
    
    model = FuzzyKNN()
    
    cv = StratifiedKFold(n_splits=5)
    
    clf = GridSearchCV(estimator=model, param_grid=parameters, 
                       cv=cv, n_jobs=-1, verbose=1, refit=True)
    
    clf.fit(X_train, y_train)
    
    best_model = clf.best_estimator_
    
    print(clf.best_estimator_)
    
    print('   evaluating model on test set')
    
    score = best_model.score(X_test, y_test)
    
    print('   ', score)
    
    logger.info(log_model(model=best_model, acc=score, dataset=f'MNIST_FULL_{digit}', serialize=True))



for digit in [0]:

    print(digit)
    
    dataset = pd.read_json(f"datasets/MNIST/digit-{digit}/mnist_full_vgg16.json")
    
    X_train = np.array(dataset.X_train.dropna().to_list())
    y_train = np.array(dataset.y_train.dropna().to_list())
    X_test = np.array(dataset.X_test.to_list())
    y_test = np.array(dataset.y_test.to_list())
    
    parameters = {'k' : [3,5,7,9,11]}
    
    model = FuzzyKNN()
    
    cv = StratifiedKFold(n_splits=5)
    
    clf = GridSearchCV(estimator=model, param_grid=parameters, 
                       cv=cv, n_jobs=-1, verbose=1, refit=True)
    
    clf.fit(X_train, y_train)
    
    best_model = clf.best_estimator_
    
    print(clf.best_estimator_)
    
    print('   evaluating model on test set')
    
    score = best_model.score(X_test, y_test)
    
    print('   ', score)
    
    logger.info(log_model(model=best_model, acc=score, dataset=f'MNIST_FULL_CNN_{digit}_pca_60_vgg16', serialize=True))



