import model as m
import numpy as np
import pandas as pd
import edit_rules as er
import preprocessing as tp
from config import ir_th

import datetime

from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler

from imblearn.pipeline import Pipeline

import argparse
import config
import json
import utils
import sys
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_val_score
import pickle
from sklearn.feature_selection import SelectKBest, chi2

from sklearn.metrics import accuracy_score, classification_report#


def clf_train(dtrain, dataset, label, classifier, transformer, seed, gridsearch):
    n_jobs=1
    
    start_time = datetime.datetime.now()
    features = dataset['features'][0]
    X = dtrain[features].str.lower() 
    y = dtrain[label].astype(int)
    if gridsearch:
        hyperparams = read_saved_hyperparms(dataset, m.get_modelName(transformer["name"], classifier["name"]), label, "hyperparameters")		
    else:
        hyperparams = None
    # Compute the imbalance ratio
    n_class, unique_classes, class_counts, ir = tp.get_class_stats(y.values)
    print('imbalance ratio', ir, 'unique_classes', unique_classes, 'class_counts', class_counts)            
    
    class_weights = {c: len(y) / (count * n_class) for c, count in zip(unique_classes, class_counts)}
    sample_weights = np.array([class_weights[class_label] for class_label in y])
    if ir > ir_th:
#             clf = Pipeline([('vect',transformer), ('clf', classifier),])
#             clf.fit(X, y)
            sampling = False
            best_model, result = set_hyperparam_search(X, y, classifier, transformer, sampling, n_jobs, seed, hyperparams)

    else:
#             clf = Pipeline([('vect',transformer), ('clf', classifier),])  #, ('sampler',sampler)
#             clf.fit(X, y, clf__sample_weight = sample_weights)            
            best_model, result = set_hyperparam_search(X, y, classifier, transformer, False, n_jobs, seed, hyperparams)
    result['ir'] = ir

    return best_model, result

def clf_test(model, dtest, label, dataset, encoder):
    """Test the ML model to predict the label
    """
    features = dataset['features'][0]
    x_test = dtest[features].str.lower() 
    y_pred = model.predict(x_test)
    #predictions
    labels_gs = dtest[f"{label}_gs"].values
    #probability distr.
    dist_proba = model.predict_proba(x_test)
    
    # save the old class values into a new column
    dtest[label + '_orig'] = dtest[label]
#    dtest[label] = y_pred
    
    # decode the label to the original classes
    #dtest = tp.decode(encoder, dtest, label)   
    
    # compute the accuracy btw predicted & correct values
    if len(encoder)>0:
        y_gs = dtest[f"{label}_gs"].map(encoder)
        dtest[label]= tp.decode(encoder, label, y_pred)
        accuracy = accuracy_score(y_gs, y_pred)
    else:
        accuracy = accuracy_score(dtest[f"{label}_gs"], y_pred)
        dtest[label]= y_pred

    return y_pred, dist_proba, dtest, accuracy
    
def get_cleaner_train_version(dataName, label, train_data, partial_key):
    if dataName == "trials_design":     
        for i in range(len(er.trials_rules)):
            if label in list(er.trials_rules[i].keys()):
                train_data = train_data[~(train_data[list(pd.Series(er.trials_rules[i]).keys())].eq(pd.Series(er.trials_rules[i]), axis=1).all(axis=1))].copy()
    train_data.reset_index(drop=True, inplace=True)    
    
    grouped = train_data.groupby(partial_key).agg(set)
    grouped[label + '_conflicted'] = grouped[label].apply(lambda x: 1 if len(x) > 1 else 0 )
    inconsistent_indices = grouped[grouped[label + '_conflicted'] == 1].index
    print('inconsistencies related to:', label, len(inconsistent_indices))
    dtrain = train_data[~(train_data[partial_key].isin(inconsistent_indices))].copy()
    
    return dtrain


def set_hyperparam_search(X_train, y_train, classifier, transformer, sampling, n_jobs=1, seed=1, hyperparams=None):
    np.random.seed(seed)
    grid_param_seed, grid_train_seed = np.random.randint(1000, size=2)
    fixed_params = classifier["fixed_params"]
    if "parallelable" in classifier.keys() and classifier['parallelable']:
        fixed_params["n_jobs"] = n_jobs

    if hyperparams is not None:
        if "hyperparams_type" in classifier and classifier["hyperparams_type"] == "int":
            hyperparams[classifier["hyperparams"]] = int(hyperparams[classifier["hyperparams"]])
        fixed_params.update(hyperparams)

    clf = classifier["fn"](**fixed_params)
    trans = transformer["fn"](**transformer["fixed_params"])
    sampler = RandomOverSampler()
    if not sampling:
        estimator = Pipeline([('vect',trans), ('clf', clf),])
    else:
        estimator = Pipeline([('vect',trans), ('sampler',sampler), ('clf', clf),])        

    # hyperparameter search
    if "hyperparams" not in classifier.keys() or hyperparams is not None:
        # if no hyper parmeter, train directly
        best_model, result = train(X_train, y_train, estimator, None, n_jobs=n_jobs, seed=grid_train_seed, skip=(hyperparams is not None))
    else:
        # grid search
        param_grid = get_param_grid(classifier, grid_param_seed, n_jobs, len(set(y_train)))
        best_model, result = train(X_train, y_train, estimator, param_grid, n_jobs=n_jobs, seed=grid_train_seed)
		
        # convert int to float to avoid json error
        if classifier["hyperparams_type"] == "int":
            result['best_params'][classifier["hyperparams"]] *= 1.0

    return best_model, result

def parse_searcher(searcher):
    """Get results from gridsearch

    Args:
        searcher: GridSearchCV object
    """
    train_accs = searcher.cv_results_['mean_train_score']
    val_accs = searcher.cv_results_['mean_test_score']
    best_idx = searcher.best_index_ 
    best_params = searcher.best_params_
    train_acc, val_acc = train_accs[best_idx], val_accs[best_idx]
    best_model = searcher.best_estimator_
    return best_model, best_params, train_acc, val_acc

def train(X_train, y_train, estimator, param_grid, seed=1, n_jobs=1, skip=False):
    """Train the model 
        
    Args:
        X_train (pd.DataFrame): features (train)
        y_train (pd.DataFrame): label (train)
        estimator (sklearn.model): model
        param_grid (dict): hyper-parameters to tune
        seed (int): seed for training
        n_jobs (int): num of threads
    """
    np.random.seed(seed)

    # cleamml
    if skip:
        best_model = estimator
        best_model.fit(X_train, y_train)
        result = {}
        return best_model, result
    
    n_class, unique_classes, class_counts, ir = tp.get_class_stats(y_train.values)    
    class_weights = {c: len(y_train) / (count * n_class) for c, count in zip(unique_classes, class_counts)}
    sample_weights = np.array([class_weights[class_label] for class_label in y_train])

    print('param_grid', param_grid)
    
    # train and tune hyper parameter with 5-fold cross validation
    if param_grid is not None:
        print('searsh for param_grid', param_grid)		
        searcher = GridSearchCV(estimator, param_grid, cv=2, n_jobs=n_jobs, return_train_score=True, scoring='accuracy', verbose=10) #, verbose=10
        searcher.fit(X_train, y_train, clf__sample_weight = sample_weights)
        best_model, best_params, train_acc, val_acc = parse_searcher(searcher)
    else: 
        print('no need')				
        # if no hyper parameter is given, train directly
        best_model = estimator
        val_acc = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy').mean()
        best_model.fit(X_train, y_train, clf__sample_weight = sample_weights)
        train_acc = best_model.score(X_train, y_train)
        best_params = {}

    result = {"best_params": best_params, "train_acc":train_acc, "val_acc": val_acc}
    return best_model, result

def get_param_grid(model, seed, n_jobs, n_class):
    """Get hyper parameters (coarse random search) """
    np.random.seed(seed)
    param_grid = {}
    
    for hp in range(len(model['hyperparams'])):
        if model["hyperparams_type"][hp] == "real":
            low, high = model["hyperparams_range"][hp]
            param_grid[model['hyperparams'][hp]] =  10 ** (-np.random.uniform(low, high, 3)) #model["hyperparams_range"][hp]  #

        if model["hyperparams_type"][hp] == "int":
            if model["name"] == "knn_classification":
                high = min(high, int(N/5*4))
        
            if len(model["hyperparams_range"][hp]) == 2 : 
                low, high = model["hyperparams_range"][hp]
                nb = 3
            else: low, high, nb = model["hyperparams_range"][hp]
            param_grid[model['hyperparams'][hp]] = model["hyperparams_range"][hp]  #np.random.randint(low, high, nb)

        if "objective" in model["hyperparams"][hp]:
            if n_class == 2: 
                param_grid[model['hyperparams'][hp]] = ['binary:logistic']
            if n_class > 2:
                param_grid[model['hyperparams'][hp]] = ['multi:softmax']

    print(param_grid)

def read_saved_hyperparms(dataset, model_name, label, param):
    """Read the parameters for the best model """
    para = {}	
    f = open(f"./results/{dataset['data_dir']}/results_training_ml.json")
    records = json.load(f)

    if param == "hyperparameters":
        para = records["with_constraints"][model_name][label]["best_params"]		
    if param == "encoder":
        if 'encoder' in records["with_constraints"][model_name][label]:
            para = records["with_constraints"][model_name][label]['encoder'] 
    f.close()
    return para
    return param_grid
