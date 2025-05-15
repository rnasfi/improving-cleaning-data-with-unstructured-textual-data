import model as m
import dataset as dt
import edit_rules as er
from config import result_dir, ir_th, root_seed, n_resplit, n_retrain, compensators

import numpy as np
import pandas as pd
import os
import json
import pickle
import datetime
import logging

from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.feature_selection import SelectKBest, chi2
import pickle

class ML_trainer:
    def __init__(self, dataset, label, lang, alg, encoder=None, parker=False):
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
        self.label = label
        self.model_name = m.get_modelName(lang, alg)
        self.transformer = m.text_transformer(lang)
        
        self.classifier = m.select_classifier(alg)
        self.estimator = None
        self.encoder = encoder # label encoder dictionary
        self.seed = root_seed
        self.n_jobs = n_retrain
        self.cv = n_resplit
        self.ir = 1
        self.imb_params = {}        

        self.params_filename = f"./{result_dir}/{self.data_name}/{self.label}_results_training_ml.json"
        if parker:
            self.strategy = "with_parker"
        else: 
            self.strategy =  "with_constraints"

        self.start_time = None
        self.confidence = 0.0

    def train(self, data, gridsearch=False):
        self.start_time = datetime.datetime.now()

        if self.encoder is not None:
            data[self.label].map(self.encoder)
            logging.info(" label training values %s", data[self.label].unique())

        
        logging.debug("read data for training %d", data.shape)
        dtrain, dvalid = dt.train_test_split(data, self.label)
        logging.debug("split the data into train (%s) and valid (%s) data ", dtrain.shape, dvalid.shape)
        
        dtrain =  self.get_cleaner_train_version(dtrain)                   
        logging.info("selected data train: %s", dtrain.shape)
        
        batch_size = dtrain.shape[0]
        # Loop through DataFrame in chunks without numpy
        for i in range(0, len(dtrain), batch_size):
            if i + batch_size < len(dtrain): batch = dtrain[i:i + batch_size]
            else: batch = dtrain[i:len(dtrain)]

            y = batch[self.label].astype(int)
            X = batch[self.features].str.lower() 
            
            # first training
            if i == 0:            
                best_model, result = self.set_hyperparam_search(X, y, gridsearch)
            else:
                print(f"Batch {(i // batch_size) + 1}:\n")
                best_model.fit(X, y)
 
        # update the imbalance ratio and save it for reporting ( future analysis?)
        logging.debug("imbalance ratio is updated and saved in this training phase %d", self.ir)

        self.confidence = self.compute_confidence_thershold(dvalid, best_model, compensators[self.data_index])
 
        self.save_model(best_model)
        # json.dump(result, open(self.params_filename, 'w'), indent=4)
        self.save_parameters(compensators[self.data_index], result)

        return best_model, result


    def define_estimator(self):
        '''
		Results:
			estimator (sklearn.Pipline): ML model
        '''
        # if gridsearch:
        #     hyperparams = self.read_saved_hyperparms("hyperparameters")		
        # else:
        #     hyperparams = None

        sampler = RandomOverSampler()
        trans = self.transformer["fn"](**self.transformer["fixed_params"])

        hyperparams = self.read_saved_hyperparms("hyperparameters")
        if hyperparams is None:        
            logging.warn("no hyperparameters found in the file")

        fixed_params = self.classifier["fixed_params"]
        if "parallelable" in self.classifier.keys() and self.classifier['parallelable']:
            fixed_params["n_jobs"] = n_jobs

        print(self.imb_params)
        if hyperparams is not None:
            if "hyperparams_type" in self.classifier and self.classifier["hyperparams_type"] == "int":
                hyperparams[self.classifier["hyperparams"]] = int(hyperparams[self.classifier["hyperparams"]])

            fixed_params.update(hyperparams)
        if fixed_params is not None: 
            logging.info("fixed_parameters updated: \n %s", fixed_params)    
        clf = self.classifier["fn"](**fixed_params)

        # should decide btw sampler or sample weight
        return Pipeline([('vect',trans), ('clf', clf),]) 

        # if self.ir > ir_th:
        #     return Pipeline([('vect',trans), ('clf', clf),])
        # else:
        #     return Pipeline([('vect',trans), ('sampler',sampler), ('clf', clf),])

    #==================================================
    # Sample selection for training
    #==================================================
    def get_cleaner_train_version(self, train_data):
        # remove nan values
        train_data.dropna(subset=[self.partial_key, self.features, self.label], inplace=True)

        # selection rules validation
        if self.data_name == "trials_design":     
            for i in range(len(er.trials_rules)):
                if self.label in list(er.trials_rules[i].keys()):
                    train_data = train_data[~(train_data[list(pd.Series(er.trials_rules[i]).keys())].
                        eq(pd.Series(er.trials_rules[i]), axis=1).all(axis=1))].copy()
        train_data.reset_index(drop=True, inplace=True)    
        
        # functional dependency valiation
        grouped = train_data.groupby(self.partial_key).agg(set)
        grouped[self.label + '_conflicted'] = grouped[self.label].apply(lambda x: 1 if len(x) > 1 else 0 )
        inconsistent_indices = grouped[grouped[self.label + '_conflicted'] == 1].index
        logging.info('inconsistencies related to %s = %d', self.label, len(inconsistent_indices))
        dtrain = train_data[~(train_data[self.partial_key].isin(inconsistent_indices))].copy()
        
        return dtrain


    #==================================================
    # Compute the hyperparameters + fine-tuning them
    #==================================================
    def set_hyperparam_search(self, X_train, y_train, gridsearch=False):	    
        grid_param_seed, grid_train_seed = np.random.randint(1000, size=2)
        sample_weights = self.compute_sample_weights(y_train.values)
        self.estimator = self.define_estimator()

        # hyperparameter search
        if "hyperparams" not in self.classifier.keys() or not gridsearch:
            # if no hyper parmeter, train directly
            best_model, result = self.fine_tune(X_train, y_train, None, sample_weights, skip=(gridsearch == False))
            logging.info("no need for fine tuning the hyperparameters!")
        else:
            # grid search
            param_grid = self.get_param_grid(self.classifier, grid_param_seed, self.n_jobs, len(set(y_train)))
            best_model, result = self.fine_tune(X_train, y_train, param_grid, sample_weights, skip=(gridsearch == False))
            logging.info("Fine tuning is done! ")
			
            # convert int to float to avoid json error
            if self.classifier["hyperparams_type"] == "int":
                result['best_params'][self.classifier["hyperparams"]] *= 1.0
        
        logging.info("result after training: \n %s", result)
        return best_model, result

    def fine_tune(self, X_train, y_train, param_grid, sample_weights, skip=True):
        """Train the model by fine-tuning the hyperparameters
	        
	    Args:
	        X_train (pd.DataFrame): features (train)
	        y_train (pd.DataFrame): label (train)
	        param_grid (dict): hyper-parameters to tune
	        sample_weights (array[int]): weight for each sample in the trainig data
        """
        np.random.seed(self.seed)
        # inspired from cleanML
        if skip:
            best_model = self.estimator
            best_model.fit(X_train, y_train)
            result = {}
            return best_model, result

        print('param_grid', param_grid)
        print('---------------------------')

        print('sample_weights', len(sample_weights))
        print('---------------------------')
	    
        # train and tune hyper parameter with 5-fold cross validation
        if param_grid is not None:
            searcher = GridSearchCV(self.estimator, param_grid, cv= self.cv, n_jobs=self.n_jobs, 
            	return_train_score=True, scoring='accuracy', verbose=10, error_score='raise')            
            searcher.fit(X_train, y_train, clf__sample_weight = sample_weights)
            best_model, best_params, train_acc, val_acc = self.parse_searcher(searcher)
        else: 
            # if no hyper parameter is given, train directly
            best_model = self.estimator
            val_acc = cross_val_score(best_model, X_train, y_train, cv= self.cv, scoring='accuracy').mean()
            best_model.fit(X_train, y_train, clf__sample_weight = sample_weights)
            train_acc = best_model.score(X_train, y_train)
            best_params = {}
        result = {"best_params": best_params, "train_acc":train_acc, "val_acc": val_acc}

        logging.info(" ML model is fine-tuned!")
        
        return best_model, result

    def parse_searcher(self, searcher):
        """
	    	Get results from gridsearch

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

    def get_param_grid(self, model, seed, n_jobs, n_class):
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
            else: 
                low, high, nb = model["hyperparams_range"][hp]
                param_grid[model['hyperparams'][hp]] = model["hyperparams_range"][hp]  #np.random.randint(low, high, nb)

            if "objective" in model["hyperparams"][hp]:
                if n_class == 2: 
                    param_grid[model['hyperparams'][hp]] = ['binary:logistic']
                if n_class > 2:
                    param_grid[model['hyperparams'][hp]] = ['multi:softmax']

            return param_grid

    # ==================================================
    # Compute the average confidence of the ML model
    # ==================================================
    def compute_confidence_thershold(self, valid_data, best_model, lambdac):
        '''
            Compute the confidence treshold to be used later in repair decision
        '''
        X_valid = valid_data[self.features]
        y_valid = valid_data[self.label]

        avg_confidence = []

        # predict the attribute values
        y_pred = best_model.predict(X_valid.str.lower())
        # predict the probability distribution for each prediction
        outputs = best_model.predict_proba(X_valid.str.lower())

        try:
            for i in range(len(set(y_valid))):# iterate over the number of classes
                    probabilities = outputs[:, i]  # Probabilities for the positive class (class 1)
                    outputs = best_model.predict_proba(X_valid.str.lower())
                    acf = self.avg_conf_correct_pred(y_valid, y_pred, probabilities, i)
                    avg_confidence.append(acf)
        except IndexError:
            logging.error("outputs: %s - y_valid: %s", outputs.shape, len(set(y_valid)))

        conf_avg = lambdac * sum(avg_confidence)/len(avg_confidence)
        logging.info('avg proba %f', conf_avg)
        return conf_avg

    def avg_conf_correct_pred(self, y_orig, y_pred, proba, v_class):
        correct_indices = (y_orig == y_pred) & (y_orig == v_class)   
        correct_probs = proba[correct_indices]

        if len(correct_probs) > 0:
            return np.mean(correct_probs)
        else: return 0.0

	# ==================================================
	# Compute the imbalance ratio and the sample weights
	# ==================================================
    def compute_sample_weights(self, y):
        '''
			Args:
				y (array or series): the value of the label to be predicted		
			Results:
				sample_weights (array):

		'''
        # Compute the imbalance ratio
        self.imb_params = self.compute_imbalance_ratio(y)  
		
        logging.info("imbalance ratio: %s- unique_classes: %s - class_counts: %s ", self.imb_params['ir'], 
        	self.imb_params['unique_classes'], self.imb_params['class_counts'])
		
        class_weights = {c: len(y) / (count * self.imb_params['n_class']) for c, count in zip(self.imb_params['unique_classes'], 
        	self.imb_params['class_counts'])}
	    
        sample_weights = np.array([class_weights[class_label] for class_label in y])

        return sample_weights

    def compute_imbalance_ratio(self, y):
        '''
			Args:
				y (array or series): the value of the label to be predicted		
			Results:
				dictioary:
				{
					'n_class':number of classes, 
					'unique_classes': the set of unique values, 
					'class_counts': the counts of each unique value, 
					'ir': imbalance ratio
				}
        '''		
        n_class, unique_classes, class_counts, ir = self.dataset.get_class_stats(y)
        self.ir = ir
        print('imbalance ratio', ir, 'unique_classes', unique_classes, 'class_counts', class_counts)
        print('---------------------------')
        return {'n_class':n_class, 'unique_classes':unique_classes, 'class_counts':class_counts, 'ir': self.ir}

	# ==================================================
	# Persist the trained ML model
	# ==================================================
    def save_model(self, model):
        # save the model
        file_model_name = f"./models/_{self.label}_classifier_{self.model_name}_{self.strategy}.pth"
        pickle.dump(model, open(file_model_name, "wb"))
        print("model saved in ", file_model_name)
        print('---------------------------') 


    def save_parameters(self, lambdac, training_hyperparameters):
        # dump json object or append it to existing one
        if os.path.exists(self.params_filename):
            with open(self.params_filename, "r") as outfile:
                try:
                    records = json.load(outfile)
                    record_per_strategy = records[self.model_name]
                except json.JSONDecodeError:
                    records = {} # Handle empty or invalid JSON
                    record_per_strategy = {}
            outfile.close()           
        else:
            records = {}
            record_per_strategy = {}

        
        logging.info("Training %s", self.strategy)
        
        # lambda coefficient
        record_per_strategy['lambda'] = lambdac

        record_per_strategy['ir'] = self.ir

        # initiate to save the parameters of the training phase
        record_per_strategy[self.strategy] = {}

        # persist the label encoder
        if self.encoder is not None:
            record_per_strategy[self.strategy]['encoder'] = self.encoder
        
        # optimal hyperparameters
        record_per_strategy[self.strategy]['best_params'] = training_hyperparameters

        # average confidence threshold
        record_per_strategy[self.strategy]['proba'] = self.confidence
        print("conf", record_per_strategy[self.strategy]['proba'])

        # training duration
        current_time = datetime.datetime.now()
        record_per_strategy[self.strategy]['duration'] = (current_time - self.start_time).total_seconds()

        records[self.model_name] = record_per_strategy # for specific model
        
        with open(self.params_filename, "w") as outfile:
            json.dump(records, outfile)

        outfile.close()


    def read_saved_hyperparms(self, param):
        """
        Read the parameters for the best model """

        param_grid = {}

        if os.path.exists(self.params_filename):
            with open(self.params_filename, "r") as outfile:
                try:
                    records = json.load(outfile)
                    logging.debug("parameters file read successfully")
                    
                    if param == "hyperparameters":
                        try:
                            param_grid = records[self.model_name][self.strategy]["best_params"]
                        except KeyError:
                            param_grid = {}

                    if param == "encoder":
                        try:
                            if 'encoder' in records[self.model_name][self.strategy]:
                                param_grid = records[self.model_name][self.strategy]['encoder']
                        except KeyError:
                            param_grid = {}

                    outfile.close()

                except json.JSONDecodeError:
                    records = {} # Handle empty or invalid JSON              
        
        return param_grid

    def train_nn():
        # ML model based on BERT and Neural Network
        model, epochs = nn.NN_train(dtrain, label, ir, unique_classes, bert_model_name, learning_rate, num_epochs, features, config.root_seed)
        result_per_label['epochs'] = epochs
        # save the ML-based pipeline
        torch.save(model, file_model_name)
