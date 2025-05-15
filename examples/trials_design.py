import os
import logging
import argparse
import random
import config

# cutomized libraries
import model as mdl
import dataset as dt
# from agents import evaluator as evals
from agents import repairer as rep
from agents import ML_trainer as mlt

def set_dir(folder_dir=None):
    """Set a new directory or path given dataset

    Args:
        folder (string): raw/missing_values/outliers/duplicates/inconsistency/mislabel
    """
    if not os.path.exists(folder_dir):
    	os.makedirs(folder_dir)
    return folder_dir

set_dir("logs")
set_dir("results")
set_dir("models")
#Configure logging 
logging.basicConfig(filename='./logs/trials_design_logs.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.basicConfig(filename='./logs/trials_design_logs.log', level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logging.basicConfig(filename='./logs/trials_design_logs.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.basicConfig(filename='./logs/trials_design_logs.log', level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == '__main__':
	logging.info("\n---------Start cleaning trials design dataset-----------")
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
	data_index = 0
	dataset = config.datasets[data_index]
	partial_key = dataset['keys'][0]
	labels = dataset['labels']
	label = labels[random.randint(0, len(labels) - 1)] #'arms' #
	feature = dataset['features'][0]
	dataName = dataset['data_name']

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

	print('cleaned with Parker:', args.parker)
	print('---------------------------')	
	print('model:', model_name)
	print('---------------------------')


	# load train & test data
	dtt = dt.Dataset(data_index, args.parker)
	dtrain, enc = dtt.read_train_csv()
	dtest = dtt.read_test_csv()

	# # # train ML model
	for l in labels:
		mltt = mlt.ML_trainer(dtt, l, trans["name"], alg["name"], enc[l], args.parker)
		try:
		    file_model_name = f"./models/_{l}_classifier_{mltt.model_name}_{mltt.strategy}.pth"
		    with open(file_model_name, 'rb') as f:
		        estimator = pickle.load(f)
		        f.close()
		except (FileNotFoundError, IOError): 
		    print("Model not found")

		else: 
			estimator, train_results = mltt.train(dtrain, args.grid_search)
			print("result of training", train_results)

	print('Done with training the model')
	print('---------------------------')

	# # evaluation of repair performance	
	ev = rep.Repairer(dtt, dtest, args.parker)
	logging.info("\n---------Repair the label attribute %s-----------", label)	
	repaired = ev.test_avg_conf(label)
	# logging.info("\n---------Empirical experiment on the confidence thresholds-----------")
	# ev.test_different_thrsholds(label)
	logging.info("\n---------Evaluate the model robustness-----------")
	ev.test_model_robustness()
