import logging
import argparse
import random
import config

# cutomized libraries
import model as mdl
import dataset as dt
from agents import evaluator as evals
from agents import ML_trainer as mlt

#Configure logging 
logging.basicConfig(filename='./logs/allergens_logs.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.basicConfig(filename='./logs/allergens_warning_logs.log', level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logging.basicConfig(filename='./logs/allergens_debug_logs.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.basicConfig(filename='./logs/allergens_error_logs.log', level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == '__main__':
	logging.info("\n---------Start cleaning allergen dataset-----------")
	parser = argparse.ArgumentParser()
	parser.add_argument('--parker', default=False, action='store_true')  
	parser.add_argument('--seed', type=int, default=config.root_seed)
	parser.add_argument('--log', default=False, action='store_true')
	parser.add_argument('--grid_search', default=True, 
		help='decide whether to fine tune the hyperparameter of the ML model')
	parser.add_argument('--data_index', type=int, default=2,  
		help='Index of the data in the list of the datsets:{0: trials design, 1: trials poulation, 2: allergens }')
	args = parser.parse_args()

	# dataset name
	data_index = 2
	dataset = config.datasets[2]
	partial_key = dataset['keys'][0]
	labels = dataset['labels']
	label = labels[random.randint(0, len(labels) - 1)]
	feature = dataset['features'][0]
	dataName = dataset['data_dir']

	print('dataset:', dataName)
	print('---------------------------')
	print('feature', feature)
	print('---------------------------')
	print('Attribute', label)
	print('---------------------------')

	# ML model
	model_name = mdl.get_best_ml_name(data_index)
	trans = mdl.get_best_transformer(data_index)
	alg = mdl.get_best_classifier(data_index)

	print('model:', model_name)
	print('---------------------------')


	# load train & test data
	dtrain, enc = dt.read_train_csv(dataset, args.parker)
	dtest = dt.read_test_csv(dataset, args.parker)

	# train ML model
	for l in  [label]: # labels: #
		mltt = mlt.ML_trainer(data_index, l, trans["name"], alg["name"], enc, args.parker)
		estimator, train_results = mltt.train(dtrain, args.grid_search)
		print("result of training", train_results)
		print("(estimation) test accuracy", estimator.score(dtest[feature], dtest[l]))
		break

	# evaluation of repair performance	
	# ev = evals.Evaluator(dataset, label, model_name, enc, args.parker)
	# print("Save the statistics from:", ev.statFile)
	# print('---------------------------')
	# ev.test_different_thrsholds(dtest)
		
	# stats = ev.get_statistics() #'Evaluator' object has no attribute 'get_statistics'
	# # # ev.save_results()