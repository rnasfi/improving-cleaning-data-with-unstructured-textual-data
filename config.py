"""Configuration of experiment and schema"""
import dataset
import model

# =============================================================================
# Directory Configuration
# =============================================================================
data_dir = 'data' # dir storing data
result_dir = 'results' # dir saving experiment results
plot_dir = 'plot' # dir saving plots

# =============================================================================
# Experiment Configuration
# =============================================================================
root_seed = 42 # root seed for entire experiments
n_resplit = 1 # num of resplit for handling split randomness
n_retrain = 5 # num of retrain for handling random search randomness
ir_th = 0.3 

# =============================================================================
# Schema Configuration
# =============================================================================
datasets = dataset.datasets
classifiers = model.classifiers
transformers = model.transformers

classifier = model.xgb
transformer = model.tfidf