from agents import evaluator as eva
import model as mdl
# import dataset as dt


import time
import warnings
import logging
import datetime
import numpy as np

class Repairer:
    def __init__(self, dataset, data, parker=False):
        """
        dataset (dict): a set of metadata about the repaired dataset        
        """
        self.dataset = dataset
        self.data_index = self.dataset.data_index
        self.keys = self.dataset.keys
        self.partial_key = self.dataset.keys[0]
        self.labels = self.dataset.labels
        self.features = self.dataset.features
        self.data_name = self.dataset.data_name
        self.encoders = self.dataset.label_encoders
    
        self.data = data
        self.vals_orig = {}
        self.vals_gs = {}
        self.vals_pred = {}
        self.vals_repair = {}
        self.dist_probabs = {}
               
        self.parker = parker
        self.strategy = "with_constraints"
        if parker:
            self.strategy = "with_parker"

    # =========================================================
    # Reapir the label attribute based on the model predictions
    # =========================================================        
    def repair(self, label, evaluator, th, data=None):
        if data is None:
            data = self.data

        self.vals_pred[label], self.dist_probabs[label], accuracy = evaluator.test(evaluator.load_model(), data)
        print(f"{label}-auxiliary accuracy = {accuracy}")
        print("------ done prdicting ----------")

        # self.data[label] = self.vals_pred[label]
        print(f"attribute values size: {len(self.vals_pred[label])}, gs attribute values size: {len(self.vals_orig[label])}")
       
        self.vals_repair[label] = evaluator.assign_repair(self.dist_probabs[label], self.vals_orig[label], self.vals_pred[label], th)
        repaired_data = data.copy()

        try:
            repaired_data[label ] = self.dataset.decode(label, self.vals_repair[label])
        except ValueError:
            print(f"{label} repairs: {set(self.vals_repair[label])} ({len(self.vals_repair[label])})")

        print("------ done replacing ----------")
        return repaired_data


    # =========================================================
    # Reapir the label attribute based on the model predictions
    # =========================================================        
    def test_avg_conf(self, label):
        encoder = self.encoders[label] if label in self.encoders else None
        evaluator = eva.Evaluator(self.dataset, label, mdl.get_best_ml_name(self.data_index), encoder, self.parker)
        self.set_label_values(label)


        repaired_data = self.repair(label, evaluator, evaluator.avg_conf)

        metrics = evaluator.get_metrics(repaired_data[label], repaired_data[label + '_orig'], repaired_data[label + '_gs'])
        logging.info("metrics: %s", metrics)
        print("metrics:", metrics)

        return repaired_data


    # ==================================================================
    # Empirical experiment to analyze the impact of different thresholds
    # ==================================================================
    def test_different_thrsholds(self, label):
        """
            Varying the values of the thresholds 
            to observe the evolution of the repair performance

            Args:
                self (Evaluator)
            
            Results:
                statistics (dict):{metrics, errors, repairs}

        """
        statistics = {}
        print('+++++++++++++++++++++Start+++++++++++++++++++++++++++++')
        encoder = self.encoders[label] if label in self.encoders else None
        evaluator = eva.Evaluator(self.dataset, label, mdl.get_best_ml_name(self.data_index), encoder)

        self.set_label_values(label)   
        self.vals_pred[label], self.dist_probabs[label], accuracy = evaluator.test(evaluator.load_model(), self.data)
        print('self.data.shape', self.data.shape)

        thresholds = [0, evaluator.avg_conf] + [th for th in np.arange(0, 1.1, 0.2)]
        thresholds.sort()
        
        print('label', label, 'avg proba', evaluator.avg_conf, 'ths', thresholds)
        print('------ done loading ----------')
        logging.info("Start at %s", datetime.datetime.now())
            
        repairs = []
        correct_repairs = []
        precisions = []
        recalls = []
        f1s = []

        # evaluate on ground truth
        for th in  thresholds:
            y_repair = evaluator.assign_repair(self.dist_probabs[label], self.vals_orig[label], self.vals_pred[label], th)
            # statistics
            correct_repair, repair, errors = evaluator.get_stats(y_repair, self.vals_orig[label], self.vals_gs[label])
            correct_repairs.append(correct_repair)   


            repairs.append(repair)

            # metrics
            metrics = evaluator.get_metrics(y_repair, self.vals_orig[label], self.vals_gs[label])
            recalls.append(round(metrics[1],2))
            precisions.append(round(metrics[0],2))
            f1s.append(round(metrics[2],2))

            print('confidence threshold', th, )
            print('statistics: correct_repairs, repairs, errors', metrics)


        self.statistics = {"errors": errors, "avg_proba": evaluator.avg_conf,\
                    "threshold": [round(th,2) for th in thresholds], 
                    'repairs': repairs, 'correct_repairs': correct_repairs, 
                    "precision": precisions, "recall": recalls, "F-1": f1s}
        print(f"+++++++++++++++++++++done with {label}+++++++++++++++++++++++++++++")
        print()

        evaluator.save_statistics(self.statistics, label)
        return self.statistics


    def test_model_robustness(self):
        """
        Returns: 
            statistics (dict): a set of metrics
        """
        statistics = {}

        # source = self.keys[1]      
        cols = [self.partial_key, self.features, self.keys[1]]
        
        sources = self.data[self.keys[1]].value_counts().keys()
        print('Sources', list(sources))

        print('+++++++++++++++++++++Start+++++++++++++++++++++++++++++')
        # Suppress all warnings 
        warnings.filterwarnings("ignore")
        
        # each time the nb of samples is increased 
        for s in range(len(sources)): 
            temp_data = self.data[self.data[self.keys[1]].isin(sources[:s+1])]
            print('current test data:', temp_data.shape, sources[:s+1])

            statistics['sources'] = list(sources[:s+1])
            statistics[s] = {}
            for ind, a in enumerate(self.labels):            
                print('label',a)
                # save appart the original values
                self.set_label_values(a, temp_data)

                # load saved model
                encoder = self.encoders[a] if a in self.encoders else None
                evaluator = eva.Evaluator(self.dataset, a, mdl.get_best_ml_name(self.data_index), encoder, self.parker)
                repaired_data = self.repair(a, evaluator, evaluator.avg_conf, temp_data)
                print(f"threshold = {evaluator.avg_conf}")

                metrics = evaluator.get_metrics(repaired_data[a].values, repaired_data[a + '_orig'].values, repaired_data[a + '_gs'].values)
                metrics1 = evaluator.get_metrics(self.vals_repair[a], self.vals_orig[a], self.vals_gs[a])

                print(f"------ done repairing {a} ({s} sources) ----------")

                # time.sleep(0.5)

                attrs = self.labels[:ind+1]

                # compute the repair metrics for the                 
                statistics[s][ind + 1] = {}
                statistics[s][ind + 1]['attributes'] = attrs
                statistics[s][ind + 1]['correct_repairs'] = self.get_all_stats(attrs)[0]
                statistics[s][ind + 1]['repairs'] = self.get_all_stats(attrs)[1]
                statistics[s][ind + 1]['errors'] = self.get_all_stats(attrs)[2]
                print('metrics', metrics, ' vs ', metrics1)
                print('stats', self.get_all_stats(attrs))

                logging.info("")
                logging.info("repair metrics for %s (data size: %s)", attrs, temp_data[attrs].shape)
                logging.info(statistics)
                print(f"+++++++++++++++++++++done with {a} ({ind + 1})+++++++++++++++++++++++++++++")
                print()

        #         # if ind > 1 : break             
            print('+++++++++++++++++++++more sources+++++++++++++++++++++++++++++')
            print()
            # break
        print(statistics)

        evaluator.save_statistics(statistics, 'nb_sources')
        return statistics 

    def get_all_stats(self, attributes):
        '''
            Args:
                attributes (list)
            Results:
                correct_repairs (list)
                repairs (list)
                errors (list)
        '''
        repairs = 0
        errors = 0
        correct_repairs = 0 
        for a in attributes:
            new_correct_repairs, new_repairs, new_errors =  self.get_stats(a)
            correct_repairs += new_correct_repairs
            repairs += new_repairs
            errors += new_errors
        return correct_repairs, repairs, errors


    def get_stats(self, label):
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
        
        for i in range(len(self.vals_repair[label])):
            if self.vals_repair[label][i] != self.vals_orig[label][i]: repairs += 1
            if self.vals_orig[label][i] != self.vals_gs[label][i]: 
                errors += 1
            if self.vals_repair[label][i] != self.vals_orig[label][i] and self.vals_repair[label][i] == self.vals_gs[label][i]: correct_repairs += 1
            #if self.vals_repair[label][i] == self.vals_gs[label][i]: print(i)

        return correct_repairs, repairs, errors

    # ==================================================================
    # Set the dataset and the values to evaluate the repair performance
    # ==================================================================
    def set_label_values(self, label, data=None):


        # test repaired by parker do not have the following columns: need to fix it!!
        if label + '_gs'not in self.data.columns:
            self.data = self.data.merge(self.dataset.read_gs_csv()[[self.partial_key, label]], 
                                  how='inner', on=self.partial_key, suffixes=('', '_gs'))
        # make a copy in the dataset about the label "original" values
        if label + '_orig' not in self.data.columns:
            self.data = self.data.merge(self.data[[self.partial_key, label]], 
                                  how='inner', on=self.partial_key, suffixes=('', '_orig')) 
        # print('current columns:', self.data.columns)
        
        if data is None:
            data  = self.data        

        # assign `self.vals_orig' the "first originally" attribute values and `self.vals_gs' the "ground truth" values
        # by using "encode()" function, we check whether thes two list of values should be converted numerically or not
        self.vals_orig[label], self.vals_gs[label] = self.dataset.encode(label, data)
        # print('orig + gs', self.vals_orig[label], self.vals_gs[label])

        # self.vals_orig[label] = data[label]
        # self.vals_gs[label] = data[label + '_gs']
        # print(f"{label}-encoder: {self.encoders[label]}")
        # print("------ done encoding (if needed) ----------")           