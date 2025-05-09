import pandas as pd
import logging
import preprocessing as tp
from sklearn.model_selection import train_test_split as tts

import csv
import os

path = os.path.abspath(os.getcwd())

def get_datasetSchema(dataName):
    """ overview about the structure of the dataset
    Parameters:
        dataName (str): name represeting the dataset
    Returns:
        if only if recognized dataName.
        dataFolder (str): relative path of the location of the dataset (.csv file)
        labels (List): list of attributes subject to prediction
        features(str/List): attribute(s) used as feature(s) to predict the labels (currently working only on a set of texts)
        keys (List): list of attributes that function as a candidte keys of the dataset (if assumed as relational data)
    """
    if dataName == 'trials_design':
        dataFolder = 'trials_design'        
        labels = ['parallel_group', 'crossover', 'randomised', 'controlled', 'open', 'double_blind', 'single_blind', 'arms' ]
        features = 'title'
        keys = ['eudract_number', 'protocol_country_code']
        return dataFolder, labels, features, keys
    
    if dataName == 'trials_population':
        dataFolder = 'trials_population'
        labels = ['elderly',  'adults',  'adolescents',  'children',  'female',  'male', 'healthy_volunteers']
        features = 'inclusion'
        keys = ['eudract_number', 'protocol_country_code']
        return dataFolder, labels, features, keys
        
    if dataName == 'allergens':
        dataFolder = 'allergens'
        labels = ['nuts', 'milk', 'gluten', 'soy', 'peanut', 'eggs']
        features = 'ingredients'
        keys = ['code', 'source']
        return dataFolder, labels, features, keys

def create_text_from_binary(row):
    """
    Parameters:
        row (tuple/):
    Returns:
        a text composed of attributes name seperated by space??
    """
    attributes = []
    attributes = row.index[row == 1].tolist()

    return ' '.join(attributes)
    
def read_data_csv(dataName, dataRole, parker):
    """ get the dataset representing the ground truth values
    Args:
        dataName (str): refers to the name representing the dataset
        dataRole (str): should be either "gs", "train", or "test"
        parker (Boolean): indicates whether get the dataset cleaned by Parker Engine 
    Returns: gs (pandas.DataFrame) 
    """
    dataFolder, labels, features, keys = get_datasetSchema(dataName)
    partial_key = keys[0]
    
    fileName = "./data/" + dataFolder + "/" + dataName + "_golden_standard.csv"
    gs = pd.read_csv(fileName, quoting=csv.QUOTE_NONNUMERIC)
    gs = gs[[partial_key] + labels] 

    if dataRole == "gs":
        return gs       
    else:
        fileName = "./data/" + dataFolder + "/" + dataName 
        data = pd.read_csv(fileName + "_" + dataRole + ".csv", quoting=csv.QUOTE_NONNUMERIC)        
        if parker: 
            data = pd.read_csv(fileName + "_parker.csv", sep = ",", quoting=csv.QUOTE_NONNUMERIC)
#             data = pd.read_csv(fileName + "_parker_" + dataRole + ".csv", sep = ",", quoting=csv.QUOTE_NONNUMERIC)
        print('relative path', "./data/" + dataFolder, '--before delete', data.shape)
        
        if dataRole == "train":            
            data.dropna(subset= [features]+[label for label in labels], inplace=True)
            data = data[data[features].str.len() > 1].copy()
            data.drop_duplicates(subset = keys + [features], keep = 'last', inplace=True, ignore_index=True)
            data = data[~(data[partial_key].isin(gs[partial_key]))]
        
        if dataRole == "test":
            #data = data.merge(gs, how='inner', on=partial_key, suffixes=('', '_gs'))
            data = data[data[partial_key].isin(gs[partial_key])]            
    
        print('--after delete', data.shape)
        return data

def train_test_split(data, label):
        dtrain, dvalid = tts(data, test_size=0.1)
        iter = 0
        while len(dvalid[label].unique()) != len(dtrain[label].unique()):
            dtrain, dvalid = tts(data, test_size=0.1)
            iter += 1
            print(str(iter) + "th iteration")
        return dtrain, dvalid


def read_gs_csv(dataset):
    """ get the dataset representing the ground truth values
    Args:
        dataName (str): refers to the name representing the dataset
    Returns: gs (pandas.DataFrame) 
    """
    return read_data_csv(dataset["data_dir"], "gs", False)
                
def read_train_csv(dataset, parker):
    """ get the dataset representing the ground truth values
    Args:
        dataName (str): refers to the name representing the dataset
        parker (Boolean): indicates whether get the dataset cleaned by Parker Engine
    Returns: gs (pandas.DataFrame) 
    """
    dataFolder, labels, features, keys = get_datasetSchema(dataset["data_dir"]) 
    dtrain = read_data_csv(dataset["data_dir"], "train", parker)
    return tp.get_encoded_labels(dtrain, dataset)

def read_test_csv(dataset, parker):
    return read_data_csv(dataset["data_dir"], "test", parker)


def read_train_test_csv(dataset, parker):
    labelEncoders = {}
    # read training dataset
    dtrain, labelEncoders = read_train_csv(dataset, parker)
    
    if len(labelEncoders) > 0: print(labelEncoders)
    
    # read test dataset
    dtest = read_test_csv(dataset, parker)
    # encode the labels if necessary
    dtest[label] = dtest.map(labelEncoders[label])

    # Domain checks constraints   
    if dataset["data_dir"] == "trials_design":
        for label in labels:
            if label == 'arms':
                # keep rows with only valid label values
                dtrain = dtrain[(dtrain['arms'] == labelEncoders[label]['2+']) | (dtrain['arms'] == labelEncoders[label]['1'])| (df['arms'] == labelEncoders[label]['0'])]
            else: 
                dtrain = dtrain[(dtrain[label] == labelEncoders[label]['Yes']) | (dtrain[label] == labelEncoders[label]['No'])]

    return dtrain, dtest, labelEncoders


Trials_design = {
    "data_dir": 'trials_design',
    "error_types": ["inconsistencies"],
    "labels":['open', 'arms', 'double_blind', 'single_blind', 'controlled', 'parallel_group', 'crossover', 'randomised'],
    "class_counts":[2,  3,  2,  2,  2,  2, 2, 2],
    "ml_task": "classification",
    "features":["title"],
    "keys":['eudract_number', 'protocol_country_code']
}

Trials_population = {
    "data_dir": 'trials_population',
    "error_types": ["inconsistencies"],
    "labels":['elderly',  'adults',  'adolescents',  'children',  'female',  'male', 'healthy_volunteers'],
    "class_counts":[2,  2,  2,  2,  2,  2, 2],
    "ml_task": "classification",
    "features":["inclusion"],
    "keys":['eudract_number', 'protocol_country_code']
}

Allergens = {
    "data_dir": 'allergens',
    "error_types": ["inconsistencies"],
    "labels":['nuts', 'milk', 'gluten', 'soy', 'peanut', 'eggs'],
    "class_counts":[3,  3,  3,  3,  3,  3],
    "ml_task": "classification",
    "features":["ingredients"],
    "keys":['code', 'source']
}

# domain of dataset 
datasets = [Trials_design, Trials_population, Allergens]