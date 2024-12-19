import random

import dataset as dt

import preprocessing as tp
import model as m
import evaluation as eva
import utils
import config
# import nn

import argparse
import logging
import pandas as pd
import numpy as np
import joblib
import datetime


import json
import csv


import sklearn
from sklearn.metrics import accuracy_score, classification_report, recall_score, precision_score

#Configure logging 
logging.basicConfig(filename='evaluations_logs.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


seed = 42

# nb of folds
splits = 3


from evaluator import evaluator as evals

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_index', type=int, default=2)
	parser.add_argument('--parker', default=False, action='store_true')  
	parser.add_argument('--seed', type=int, default=1)

	parser.add_argument('--log', default=False, action='store_true')     

	args = parser.parse_args()

	# dataset names
	data_index = args.data_index
	if data_index == 2 : 
		j = 0
	else: 
		j = 1
	dataset = config.datasets[args.data_index] #dt.Allergens
	partial_key = dataset['keys'][0]
	labels = dataset['labels']
	feature = dataset['features'][0]
	dataName = dataset['data_dir']

	print('dataset:', dataName)
	print('---------------------------')
	print('feature', feature)
	print('---------------------------')
	print('labels', labels)
	print('---------------------------')

	#LEARNING ALGORITHM
	alg = config.classifier #'multinomial bayesian' #'xgboost'
	#LANGUAGE MODEL
	lang = config.transformers[j] #'tf-idf'
	#MODEL
	model_name = m.get_best_ml(data_index)
	print('model:', model_name)
	print('---------------------------')

	ev = evals.Evaluator(dataset, args.parker, model_name)

	stats = ev.get_statistics()

	ev.save_results()
	ev.encoders = ev.get_data_encoders()