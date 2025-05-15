import os
import pickle
import logging
import warnings

import json
import numpy as np
import csv

from config import result_dir
import model as m

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
        self.data_index = self.dataset.data_index
        self.keys = self.dataset.keys
        self.partial_key = self.dataset.keys[0]
        self.labels = self.dataset.labels
        self.features = self.dataset.features
        self.data_name = self.dataset.data_name

        self.label = attribute

        self.model_name = model_name
        self.params_filename = f"./{result_dir}/{self.data_name}/{self.label}_results_training_ml.json"
        self.avg_conf = None
        
        self.data = None
        self.encoder = encoder
        
        self.strategy = "with_constraints"
        if parker:
            self.strategy = "with_parker"
        self.model_parameters = self.load_training_parameters()

        self.statFile = f"./{result_dir}/{self.data_name}/{self.data_name}_stats_{self.model_name}.json"
        #trials_design_stats_tf-idf-xgboost_with_parker

    # ========================
    # Predict the label values
    # ========================   
    def test(self, model, data):
        """
            Test the ML model to predict the label
            Results:
                y_pred (1D np.array)
                outputs (multidimension np.array)
                accuracy (1D np.array)
        """
        logging.info("read data for testing %s", data.shape)
        X_test = data[self.features].str.lower() 
        y_pred = model.predict(X_test)
        
        #predictions
        labels_gs = data[f"{self.label}_gs"].values
        #probability distr.
        dist_proba = model.predict_proba(X_test)
        
        # save the old class values into a new column
        data[self.label + '_orig'] = data[self.label]

        # compute the accuracy btw predicted & correct values
        if self.encoder is not None and label in self.encoder:
            # convert the ground truth label values to numerical values
            y_gs = data[f"{self.label}_gs"].map(self.encoder)
            # assign the label column the predicted values and put them as originally 
            data[self.label]= self.dataset.decode(self.label, y_pred)
            accuracy = accuracy_score(y_gs, y_pred)
        else:
            accuracy = accuracy_score(data[f"{self.label}_gs"], y_pred)
            data[self.label]= y_pred

        return y_pred, dist_proba, accuracy

    def assign_repair(self, dist_proba, y_orig, y_pred, th):
        """
        dist_proba (List): 2D list of probability distribution of for each label's possible classes
        y_orig (List): list of label's real values
        th (float): threshold of certainty 
        Retruns (List[float]):
        list of label's repaired values
        """
        final_predictions = []
        for i, proba in enumerate(dist_proba):
            if np.max(proba) >= th  or np.isnan(y_orig[i]):
                final_predictions.append(y_pred[i])
            else: final_predictions.append(y_orig[i])
        return final_predictions


    # =======================================
    # Featch list of variables
    # =======================================     
    def get_vars(self, data, list_vars):
        if list_vars == self.keys[1]:
            return data[list_vars].value_counts().keys()
        if list_vars == "attributes":
            return data[label] # list of structured attributes
        return []

    # =======================================
    # Load the persistent ML model parameters
    # =======================================
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
                self.avg_conf = round(records[self.model_name][self.strategy]['proba'],2)
                
                # kind of redudant instruction - to be assessed later
                if 'encoder' in records[self.model_name][self.strategy]:
                    self.encoder = records[self.model_name][self.strategy]['encoder']  
                outfile.close()

        except json.JSONDecodeError:
            records = {} # Handle empty or invalid JSON  

        return records        

    def get_data_encoders(self):
        """
        Returns: encoders (dict): how each label to be repaired should be encoded according to the ML model
        """
        records = self.model_parameters
        if 'encoder' in records[self.model_name][self.strategy]:
                self.encoder = records[self.model_name][self.strategy]['encoder']           
        return encoder

    def save_results(self):  
        logging.info("read file: %s", self.statFile)
        with open(statFile, "w") as outfile: 
                json.dump(self.statistics, outfile)      



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

    # ============================
    # Performance statsistics
    # ============================
    def get_metrics(self, y_pred, y_orig, y_gs):
        """
        y_pred (List): list of label's predicted values
        y_orig (List): list of label's real values
        y_gs (List): list of label's ground truth values 
        Retruns (List[float]):
        prediction metrics: precision, recall and F-1 score
        """
        repairs = 0
        errors = 0
        correct_repairs = 0
        precision = 0
        recall = 0
        f1 = 0
        
        for i in range(len(y_pred)):
            if y_pred[i] != y_orig[i]: repairs += 1
            if y_orig[i] != y_gs[i]: errors += 1
            if y_pred[i] != y_orig[i] and y_pred[i] == y_gs[i]: correct_repairs += 1
        if repairs != 0: precision = round(correct_repairs/repairs, 2)
        if errors != 0: recall = round(correct_repairs/errors, 2)
        if precision != 0 or recall != 0: f1 = round(2 * (precision * recall)/(precision + recall), 2)
        print('correct repair', correct_repairs, 'repairs', repairs, 'errors', errors)
        return precision, recall, f1

    def get_stats(self, y_pred, y_orig, y_gs):
        """ Provide some statistics about the repairs per cells
        
        y_pred (List): list of label's predicted values
        y_orig (List): list of label's real values
        y_gs (List): list of label's ground truth values

        Retruns (List[int]):
        prediction statistics: number of correct repairs, number of repairs and number of erroneous cells
        """    
        repairs = 0
        errors = 0
        correct_repairs = 0    
        
        for i in range(len(y_pred)):
            if y_pred[i] != y_orig[i]: repairs += 1
            if y_orig[i] != y_gs[i]: 
                errors += 1
            if y_pred[i] != y_orig[i] and y_pred[i] == y_gs[i]: correct_repairs += 1
            #if y_pred[i] == y_gs[i]: print(i)
        
        return correct_repairs, repairs, errors


    def save_statistics(self, statistics, key):
        # dump json object or append it to existing one
        if os.path.exists(self.statFile):
            with open(self.statFile, "r") as outfile:
                try:
                    records = json.load(outfile)
                    outfile.close()
                except json.JSONDecodeError:
                    records = {} # Handle empty or invalid JSON
        else:
            records = {}

        # add object assignd for the attribute subject to repair
        # including the empirical results of thresholds
        records[key] = statistics

        logging.info("recorded labels: %s", records.keys())

        with open(self.statFile, "w") as outfile:
            json.dump(records, outfile)
            outfile.close() 
            return 0