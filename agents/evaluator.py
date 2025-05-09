import os
import pickle
import logging
import datetime
import warnings

import json
import numpy as np
import csv

from config import result_dir
import model as m
import dataset as dt
import training as tr

import evaluation as eva
import preprocessing as tp
from sklearn.metrics import accuracy_score

class Evaluator:
    def __init__(self, dataset, attribute, model_name, encoder=None, parker=False):
        """
            Initialize the evaluator agent to see the performence of the ML model in repairing the attribute value
            
            Args:
                dataset (dict): a set of metadata about the repaired dataset        
                parker (bool): if the data was already repaired by Parker engine or not
                encode (dict): map the attribute values (if not numerical) to numerical converted values
        """
        self.dataset = dataset
        self.keys = dataset['keys']
        self.partial_key = dataset['keys'][0]
        self.label = attribute
        self.features = dataset['features'][0]
        self.dataName = dataset['data_dir']
        self.parker = parker
        self.model_name = model_name

        self.data = None
        self.encoder = encoder
        
        self.params_filename = f"./{result_dir}/{self.dataName}/{self.label}_results_training_ml.json"
        self.statFile = f"./{result_dir}/{self.dataName}/{self.dataName}_stats_{self.model_name}_robustness.json"
        
        self.statistics = None
        
        if parker:
            self.strategy = "with_parker"
        else: 
            self.strategy =  "with_constraints"        
 
    def get_vars(self, data, list_vars):
        if list_vars == self.keys[1]:
            return data[list_vars].value_counts().keys()
        if list_vars == "attributes":
            return data[label] # list of structured attributes
        return []

    def load_training_parameters(self):
        """
            Results:
                records (dict): {
                    avg_proab (float): estimation of the ML model confidenc,
                    encoders (dict): mapper of the (numerically) converted label values to their original value
                    }
        """
        
        try:
            with open(self.params_filename, "r") as outfile:
                records = json.load(outfile)
                logging.debug("parameters file read successfully")
                logging.info(' load training parameers from:  %s', self.params_filename)
                logging.info(' parameters: \n %s', records)
                outfile.close()

        except json.JSONDecodeError:
            records = {} # Handle empty or invalid JSON          
        return records        

    def get_data_encoders(self):
        """
        Returns: encoders (dict): how each label to be repaired should be encoded according to the ML model
        """
        records = self.load_training_parameters()

        encoder = {} 
        if 'encoder' in records[self.model_name][self.strategy]:
                self.encoder = records[self.model_name][self.strategy]['encoder']           
        return encoder

    def save_results(self):
        
        logging.info("read file: %s", self.statFile)
        with open(statFile, "w") as outfile: 
                json.dump(self.statistics, outfile)      

    def test(self, model, data):
        """
            Test the ML model to predict the label
            Results:
                y_pred (1D np.array)
                outputs (multidimension np.array)
                accuracy (1D np.array)
        """
        logging.info("read data for testing %s", dtest.shape)
        X_test = data[self.features].str.lower() 
        y_pred = model.predict(X_test)
        
        #predictions
        labels_gs = data[f"{self.label}_gs"].values
        #probability distr.
        dist_proba = model.predict_proba(X_test)
        
        # save the old class values into a new column
        data[self.label + '_orig'] = data[self.label]

        # compute the accuracy btw predicted & correct values
        if len(self.encoder) > 0:
            y_gs = data[f"{self.label}_gs"].map(self.encoder)
            data[self.label]= tp.decode(self.encoder, self.label, y_pred)
            accuracy = accuracy_score(y_gs, y_pred)
        else:
            accuracy = accuracy_score(data[f"{self.label}_gs"], y_pred)
            data[self.label]= y_pred

        return y_pred, dist_proba, accuracy

    # ============================
    # Load the persistent ML model
    # ============================
    def load_model(self):
        # load saved model
        file_model_name = f"./models/_{self.label}_classifier_{self.model_name}_{self.strategy}.pth"
        with open(file_model_name, 'rb') as f: 
            model = pickle.load(f)
        f.close()
        return model


    # ==================================================================
    # Empirical experiment to analyze the impact of different thresholds
    # ==================================================================
    def test_different_thrsholds(self, dtest):
        """
            Varying the values of the thresholds 
            to observe the evolution of the repair performance

            Args:
                self (Evaluator)
            
            Results:

        """
        statistics = {}
        records = self.load_training_parameters()

        print('+++++++++++++++++++++Start+++++++++++++++++++++++++++++')

        self.data = dtest.copy()
        print(self.data.shape)

        # test repaired by parker do not have the following columns: need to fix it!!
        if self.label + '_gs'not in self.data.columns:
            self.data = self.data.merge(dt.read_gs_csv(self.dataset)[[self.partial_key, self.label]], 
                                  how='inner', on=self.partial_key, suffixes=('', '_gs'))

        avg_proba = round(records[self.model_name][self.strategy]['proba'],2)

        thresholds = [0, avg_proba] + [th for th in np.arange(0, 1.1, 0.2)]
        thresholds.sort()
        
        print('label', self.label, 'avg proba', avg_proba, 'ths', thresholds)
        print('------ done loading ----------')
        logging.info("Start at %s", datetime.datetime.now())

        # get the encoder if exists and encode y_orig  y_gs
        enc = {}
        enc, y_orig, y_gs = tp.encode(self.encoder, self.label, self.data)
        print('encoder', enc, self.encoder)
        print("------ done encoding ----------")
        logging.info("------ done encoding ----------")      
            
        # predict the values for the labels to be repaired
        y_pred, outputs, accuracy = self.test(self.load_model(), self.data)
        print("------ done predicting ----------")
        logging.info("------ done predicting ----------")

        # make a copy in the dataset about the original values
        if self.label + '_orig' not in self.data.columns:
            self.data = self.data.merge(self.data[[self.partial_key, self.label]], 
                                  how='inner', on=self.partial_key, suffixes=('', '_orig')) 
            print('current columns:', self.data.columns)
            
        repairs = []
        correct_repairs = []
        precisions = []
        recalls = []
        f1s = []

        # evaluate on ground truth
        for th in  thresholds: 
            y_repair = eva.assign_repair(outputs, y_orig.values, y_pred, th)
            # statistics
            correct_repair, repair, errors = eva.get_stats(y_repair, y_orig.values, y_gs.values)
            correct_repairs.append(correct_repair)
            repairs.append(repair)

            # metrics
            metrics = eva.get_metrics(y_repair, y_orig.values, y_gs.values)
            recalls.append(round(metrics[1],2))
            precisions.append(round(metrics[0],2))
            f1s.append(round(metrics[2],2))

            print('confidence threshold', th, )
            print('statistics: correct_repairs, repairs, errors', metrics)


        statistics = {"errors": errors, "avg_proba": avg_proba,\
                    "threshold": [round(th,2) for th in thresholds], 
                    'repairs': repairs, 'correct_repairs': correct_repairs, 
                    "precision": precisions, "recall": recalls, "F-1": f1s}
        print(f"+++++++++++++++++++++done with {self.label}+++++++++++++++++++++++++++++")
        print()

    def get_metrics(self):
            return 0

