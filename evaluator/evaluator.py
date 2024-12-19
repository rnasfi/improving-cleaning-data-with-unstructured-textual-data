import os
import time
import pickle
import logging
import warnings

import json
import csv

import model as m
import dataset as dt
import training as tr

import evaluation as eva
import preprocessing as tp

class Evaluator:
    def __init__(self, dataset, parker, model_name):
        """
        dataset (dict): a set of metadata about the repaired dataset        
        parker (bool): if the data was already repaired by Parker engine or not
        """
        self.dataset = dataset
        self.keys = dataset['keys']
        self.partial_key = dataset['keys'][0]
        self.labels = dataset['labels']
        self.feature = dataset['features'][0]
        self.dataName = dataset['data_dir']
        self.parker = parker
        self.model = model_name
        self.encoders = self.get_data_encoders()#{}
        
        self.stats = None
 
    def get_vars(self, data, list_vars):
        if list_vars == self.keys[1]:
            return data[list_vars].value_counts().keys()
        if list_vars == "attributes":
            return self.labels 
        return []

    def get_strategy_name(self):
        if self.parker:
            strategy_name = "with_parker"
        else: strategy_name = "with_constraints"

        return strategy_name

    def get_data_encoders(self):
        """
        Returns: encoders (dict): how each label to be repaired should be encoded according to the ML model
        """

        configFile = os.path.join('.', 'results', f"{self.dataName}/results_training_best_ml.json")

        print('configFile', configFile)

        f = open(configFile)

        records = json.load(f)

        encoder = {} 
        for label in self.labels:
            if 'encoder' in records[self.get_strategy_name()][self.model][label]:
                encoder[label] = records[self.get_strategy_name()][self.model][label]['encoder']
            
        f.close()
        return encoder

    def save_results(self):
        statFile = f"./results/{self.dataName}/{self.dataName}_stats_{self.model}_robustness.json"
        print(statFile)
        with open(statFile, "w") as outfile: 
                json.dump(self.stats, outfile)      

    def test_model_robustness(self):
        """
        Returns: 
        stats (dict): a set of metrics
        """
        stats = {}

        #read test data
        dtest = dt.read_test_csv(self.dataName, False)

        source = self.keys[1]      
        cols = [self.partial_key, self.feature, source]
        
        sources = self.get_vars(dtest,  source)
        print('Sources', sources)

        logging.info(self.dataset)
        print('+++++++++++++++++++++Start+++++++++++++++++++++++++++++')

         # Suppress all warnings 
        warnings.filterwarnings("ignore")
        
        # each time the nb of samples is increased 
        for s in range(len(sources)):
            dtest1 = dtest[dtest[source].isin(sources[:s+1])]
            print(dtest1.shape, sources[:s+1])
            stats[s] = {}
            for ind, a in enumerate(self.labels):            
                print('label',a)
                # save appart the original values
                dtest1[a + '_orig'] = dtest1[a].values

                ## load saved model
                file_model_name = os.path.join('.', 'models', \
                    f"_{a}_classifier_{self.model}_{self.get_strategy_name()}.pth")
                with open(file_model_name, 'rb') as f: model = pickle.load(f)   


                # get the encoder if exists
                enc = {}
                y_orig = dtest1[a + '_orig']
                y_gs = dtest1[a + '_gs']

                enc, y_orig, y_gs = tp.encode(self.get_data_encoders(), a, dtest1)
                print("------ done encoding ----------")

                y_pred, outputs, dtest, accuracy = tr.clf_test(model, dtest1, a, self.dataset, enc)
                print("------ done predicting ----------")

                # replace with the predicted values 
                if len(enc) > 0:
                    dtest1[a] = tp.decode(enc, a, y_pred)
                else: dtest1[a] = y_pred

                time.sleep(1)

                attrs = self.labels[:ind+1]

                # compute the repair metrics for the 
                
                stats[s][ind + 1] = {}
                stats[s][ind + 1]['correct_repairs'] = eva.get_all_stats(dtest1, attrs)[0]
                stats[s][ind + 1]['repairs'] = eva.get_all_stats(dtest1, attrs)[1]
                stats[s][ind + 1]['errors'] = eva.get_all_stats(dtest1, attrs)[2]

                print(len(sources[:s+1]), eva.get_all_stats(dtest1, attrs))
                print(attrs, 'data size', dtest1[attrs].shape)
                logging.info("")
                logging.info("repair metrics for %s (data size: %s)", attrs, dtest1[attrs].shape)
                logging.info(stats)
                print(f"+++++++++++++++++++++done with {a} ({ind + 1})+++++++++++++++++++++++++++++")
                print()

                # if ind > 1 : break             
            print('+++++++++++++++++++++more sources+++++++++++++++++++++++++++++')
            print()

        self.stats = stats
        return stats

    def test_different_thrsholds(self):
        stats = {}
        dtest = dt.read_test_csv(self.dataName, self.parker)

        print('+++++++++++++++++++++Start+++++++++++++++++++++++++++++')

        dtest1 = dtest.copy()
        print(dtest1.shape)

        for a in self.labels:
            # test repaired by parker do not have the following columns: need to fix it!!
            if a + '_gs'not in dtest1.columns:
                dtest1 = dtest1.merge(dt.read_gs_csv(self.dataName)[[self.partial_key, a ]], 
                                      how='inner', on=self.partial_key, suffixes=('', '_gs'))


            avg_proba = round(records[self.get_strategy_name()][self.model][a]['proba'],2)

            thresholds = [0, avg_proba] + [th for th in np.arange(0, 1.1, 0.2)]

            ## load saved model
            file_model_name = f"./models/_{a}_classifier_{self.model}_{_with}.pth"
            with open(file_model_name, 'rb') as f: model = pickle.load(f)   

            thresholds.sort()
            print('label',a, 'avg proba', avg_proba, 'ths', thresholds)

            # get the encoder if exists and encode y_orig  y_gs
            enc = {}
            enc, y_orig, y_gs = tp.read_encoder(self.encoders, a, dtest1)
            print("------ done encoding ----------")      
            
            # predict the values for the labels to be repaired
            y_pred, outputs, accuracy = tr.clf_test(model, dtest1, a, self.dataset, enc)
            print("------ done predicting ----------")

        #     if a + '_orig' not in dtest1.columns:
        #         dtest1 = dtest1.merge(dtest1[[self.partial_key, a ]], 
        #                               how='inner', on=self.partial_key, suffixes=('', '_orig')) 
        #         print('current columns:', dtest1.columns)
            
        #     repairs = []
        #     correct_repairs = []
        #     precisions = []
        #     recalls = []
        #     f1s = []

        #     # evaluate on ground truth
        #     for th in  thresholds: 
        #         y_repair = eva.assign_repair(outputs, y_orig.values, y_pred, th)
        #         # stats
        #         correct_repair, repair, errors = eva.get_stats(y_repair, y_orig.values, y_gs.values)
        #         correct_repairs.append(correct_repair)
        #         repairs.append(repair)

        #         # metrics
        #         metrics = eva.get_metrics(y_repair, y_orig.values, y_gs.values)
        #         recalls.append(round(metrics[1],2))
        #         precisions.append(round(metrics[0],2))
        #         f1s.append(round(metrics[2],2))

        #         print(' th', th, )
        #         print('stats: correct_repairs, repairs, errors', metrics)

        #     stats[a] = {"errors": errors, "avg_proba": avg_proba,\
        #                 "threshold": [round(th,2) for th in thresholds], 'repairs': repairs, 'correct_repairs': correct_repairs, "precision": precisions, "recall": recalls, "F-1": f1s}
        #     print(f"+++++++++++++++++++++done with {a}+++++++++++++++++++++++++++++")
        #     print()
        # #         break
        # print('+++++++++++++++++++++more sources+++++++++++++++++++++++++++++')
        # print()            
