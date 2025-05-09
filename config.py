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
root_seed = model.root_seed # root seed for entire experiments
n_resplit = 2 # num of resplit for handling split randomness
n_retrain = 5 # num of retrain for handling random search randomness
ir_th = 0.3 

# =============================================================================
# Schema Configuration
# =============================================================================
datasets = dataset.datasets # [Trials_design, Trials_population, Allergens]
classifiers = model.classifiers
transformers = model.transformers
compensators = [.9,.9,.9] # coefficient to compute the confidence threshold

classifier = model.xgb
transformer = model.tfidf
