import pandas as pd
import csv
import os
import config
import sys
import json
import numpy as np


# =============================================================================
# Data related utils
# =============================================================================

def get_dataset(name):
    """Get dataset dict in config.py given name

    Args:
        name (string): dataset name
    """ 
    dataset = [d for d in config.datasets if d['data_dir'] == name]
    if len(dataset) == 0:
        print('Dataset {} does not exist.'.format(name))
        sys.exit()
    return dataset[0]
        
def get_classifier(name):
    """Get ML classifier dict in config.py given name

    Args:
        name (string): algorithm name
    """ 
    classifier = [d for d in config.classifiers if d['name'] == name]
    if len(classifier) == 0:
        print('Dataset {} does not exist.'.format(name))
        sys.exit()
    return classifier[0]

def get_gt_dir(dataset):
    """Get golden standard data filename for a given dataset

    Args:
        dataset(dict): dataset dict in config.py
    """
    data_dir = os.path.join(config.data_dir, dataset['data_dir'])
    return os.path.join(data_dir, dataset['data_dir'] + "_golden_standard.csv")

def get_dir(dataset, folder=None, file=None, create_folder=False):
    """Get directory or path given dataset, folder name (optional) and filename (optional)

    Args:
        dataset(dict): dataset dict in config.py
        folder (string): raw/missing_values/outliers/duplicates/inconsistency/mislabel
        file (string): file name
        create_folder (bool): whether create folder if not exist
    """
    data_dir = os.path.join(config.data_dir, dataset['data_dir'])
    if folder is None and file is None:
        return data_dir

    if folder is not None:
        folder_dir = os.path.join(data_dir, folder)
    else: folder_dir = data_dir
        
    if create_folder and not os.path.exists(folder_dir):
        os.makedirs(folder_dir)

    if file is None:
        return folder_dir
    
    file_dir = os.path.join(folder_dir, file)
    return file_dir
    

def load_df(dataset, file_path):
    """load data file into pandas dataframe and convert categorical variables to string

    Args: 
        dataset (dict): dataset in config.py
        file_path (string): path of data file
    """
    df = pd.read_csv(file_path)
    if 'categorical_variables' in dataset.keys():
        categories = dataset['categorical_variables']
        for cat in categories:
            df[cat] = df[cat].astype(str).replace('nan', np.nan) 
    return df

def load_dfs(dataset, file_path_pfx, return_version=False):
    """load train and test files into pandas dataframes 

    Args:
        dataset (dict): dataset in config.py
        file_path_pfx (string): prefix of data file
        return_version (bool): whether to return the version (split seed) of data
    """
    train_dir = file_path_pfx + '_train.csv'
    test_dir = file_path_pfx + '_test.csv'
    train = load_df(dataset, train_dir)
    test = load_df(dataset, test_dir)
    if return_version:
        version = get_version(file_path_pfx)
        return train, test, version
    else:
        return train, test

def save_dfs(data, dataRole, save_path_pfx, version=None):
    """Save train and test pandas dataframes in csv file

    Args:
        train (pd.DataFrame): training set
        test (pd.DataFrame): test set
        save_path_pfx (string): prefix of save path
        version (int): version of data (optional)
    """
    data_save_path = save_path_pfx + dataRole + '.csv'
    data.to_csv(data_save_path, quoting=csv.QUOTE_NONNUMERIC, index=False)
    if version is not None:
        save_version(save_path_pfx, version)

def save_version(file_path_pfx, seed):
    """Save version of data in json file

    Args:
        file_path_pfx (string): prefix of path of data file 
        seed (int): split seed of data
    """
    directory, file = os.path.split(file_path_pfx)
    version_path = os.path.join(directory, "version.json")
    if os.path.exists(version_path):
        version = json.load(open(version_path, 'r'))
    else:
        version = {}
    version[file] = str(seed)
    json.dump(version, open(version_path, 'w'))

def get_version(file_path_pfx):
    """Get version of data 

    Args:
        file_path_pfx (string): prefix of path of data file 
    """
    directory, file = os.path.split(file_path_pfx)
    version_path = os.path.join(directory, "version.json")
    print('version_path=',version_path)
    if os.path.exists(version_path):
        version = json.load(open(version_path, 'r'))
        return int(version[file])
    else:
        return None

def remove(path):
    """Remove file or directory

    Args:
        path (string): path of file or directory
    """
    if os.path.isfile(path):
        os.remove(path) 
    elif os.path.isdir(path):
        shutil.rmtree(path)
        
# =============================================================================
# Training related utils
# =============================================================================
def check_completed(dataset, split_seed):
    """Check whether all experiments for the dataset with split_seed have been completed
    
    Args:
        datasets (dict): dataset dict in config.py
        split_seed (int): split seed
        experiment_seed (int): experiment seed
    """
    result = load_result(dataset['data_dir'])
    seeds = np.random.randint(10000, size=config.n_retrain)

    # add for each scenarios
    for model in config.classifiers:
        print('config.classifiers', model, config.classifiers, result.keys())
        key = "{}/v{}/{}".format(dataset['data_dir'], split_seed, model['name'])
        print(key, key not in result.keys())
        if key not in result.keys():
            return False
    return True

# =============================================================================
# Result related utils
# =============================================================================
def load_result(dataset_name=None, parse_key=False):
    """Load result of one dataset or all datasets (if no argument) from json to dict

    Args:
        dataset_name (string): dataset name. If not specified, load results of all datasets.
        parse_key (bool): whether convert key from string to tuple
    """
    if dataset_name is None:
        files = [file for file in os.listdir(config.result_dir) if file.endswith('_result.json')]
        result_path = [os.path.join(config.result_dir, file) for file in files]
    else:
        result_path = [os.path.join(config.result_dir, '{}_result.json'.format(dataset_name))]
    print('result_path', result_path)

    result = {}
    for path in result_path:
        if os.path.exists(path):
            result.update(json.load(open(path, 'r')))

    if parse_key:
        new_result = {}
        for key, value in result.items(): #e.g new_key : dataset_name/split_seed/model_name/seed
            new_key = tuple(key.split('/'))
            new_result[new_key] = value
        result = new_result

    return result

def save_result(dataset_name, key, res):
    """Save result to json

    Args:
        dataset_name (string): dataset name. 
        key (string): key of result in form: dataset_name/split_seed/error_type/clean_method/model_name/seed
        res (dict): result dict {metric_name: metric result}
    """
    result = load_result(dataset_name)
    result[key] = res
    result_path = os.path.join(config.result_dir, '{}_result.json'.format(dataset_name))
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)
    json.dump(result, open(result_path, 'w'), indent=4)        
        
# =============================================================================
# ML model utils
# =============================================================================
                 
