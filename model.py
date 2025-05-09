# =============================================================================
# Python libraries
# ============================================================================= 
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

"""Define the domain of ML model"""
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier

from sklearn import metrics, linear_model, tree

root_seed = 42

# =============================================================================
# ML model naming (format: concat(NLP technique, ML algorithm ) )
# ============================================================================= 
def get_modelName(lang, alg):
    model_name = lang + '-' + alg
    return model_name.replace(' ','_')

# =============================================================================
# Selection of textual transformer 
# ============================================================================= 
def text_transformer(lang):
    if lang == 'tf-idf':
        return tfidf

    elif lang == 'count-vect':
        return countvect

    else: tr = None

    return tr

# =============================================================================
# Textual transformer parameters
# =============================================================================
countvect = {
    "name":"count-vect",
    "fn": CountVectorizer,
    "fixed_params": {"strip_accents":'ascii', "stop_words":'english', "token_pattern": r"\b[a-zA-Z]+\b"}, #"strip_accents":'ascii', "stop_words":'english', "token_pattern": r"\b[a-zA-Z]+\b"
}

tfidf = {
    "name":"tf-idf",
    "fn": TfidfVectorizer,
    "fixed_params": {"strip_accents":'ascii', "stop_words":'english', "token_pattern": r"\b[a-zA-Z]+\b"},
}

# =============================================================================
# Selection of ML algorithm
# ============================================================================= 
def select_classifier(alg):
    if alg == 'logistic regression l1' or alg == 'lr':
        return lr

    if alg == 'logistic regression' or alg == 'lr2':
        return lr2

    if alg == 'decision tree' or alg == 'dc':
        return dc

    if alg == 'support vector machine' or alg == 'svc':
        return svc

    if alg == 'xgboost' or alg == 'xgb':
        return xgb

    if alg == 'multinomial bayesian' or alg == 'mnb':
        return mnb

    if alg == 'random forest':
        return rf

# =============================================================================
# ML models parameters
# =============================================================================
mnb = {
    "name": "mnb",
    "fn": MultinomialNB,
    "fixed_params": {"alpha":0.8, "fit_prior":True, "force_alpha":True},#
    "type": "classification",
#     "hyperparams": "clf__alpha" ,
#     "hyperparams_type": "real",
    #"hyperparams_range": [0.05, 0.09]
}    
# linear model    
lr = {
    "name": "logistic_regression l1",
    "fn": LogisticRegression,
    "fixed_params": {"solver":"lbfgs", "max_iter":5000, "multi_class":'auto', "penalty":"l2"},
    "parallelable": True,
    "type": "classification",
    "hyperparams": ["clf__C"] ,
    "hyperparams_type": ["real"],
    "hyperparams_range": [[-5, 5]]
}
lr2 = {
    "name": "logistic_regression", #l2
    "fn": LogisticRegression,
    "fixed_params": {"max_iter":1000, "penalty":"l2", "multi_class":'multinomial'},
    "parallelable": True,
    "type": "classification",
    "hyperparams": ["clf__C", "clf_solver"] ,
    "hyperparams_type": ["real"],
    "hyperparams_range": [[ 0.01, 1],["newton-cg", "lbfgs", "sag", "saga"]]
} 
svc = {
    "name": "support vector machine",
    "fn": LinearSVC,
    "fixed_params": {"class_weight":"balanced", "random_state=":root_seed},
    "type": "classification",
    "hyperparams": ["clf__loss", "clf_C", "clf_penalty"] ,
    "hyperparams_type": ["string", "real", "string"],
    "hyperparams_range": [['hinge', 'squared_hinge'], [ 0.01, 1], ["l2"]]           
}

# decision tree
dc = {
    "name":"decision tree",        
    "fn": tree.DecisionTreeClassifier,
    "fixed_params": {"random_state":root_seed},
    "type": "classification",
    "hyperparams": ["clf__criterion", "clf__max_depth"],
    "hyperparams_type": ["string", "int"],
    "hyperparams_range": [['entropy', 'gini'],[10, 20, 28]]    
}   
# sklearn.ensemble
xgb = {
    "name":"xgboost",
    "fn": XGBClassifier,
    "fixed_params": {}, #"clf__learning_rate": 0.1
    "type": "classification",
    "hyperparams": ["clf__max_depth", "clf__n_estimators", "clf__objective", "clf__learning_rate"],
    "hyperparams_type": ["int", "int", "string", "real"],
    "hyperparams_range": [[6, 8, 10],[200, 300, 220], None, [.1,.2]] #["binary:logistic","multi:softmax"] num_class=n_class
}
rf = {
    "name":"random forest",
    "fn": RandomForestClassifier,
    "fixed_params": {"random_state":root_seed},
    "type": "classification",
    "hyperparams": None, 
}

# =============================================================================
# Artificial Neural Network model
# =============================================================================  
""" Build ANN model by defining: number of neurons; optimizer function; loss function
    inputs (tensor): the (textual) inputs
    Returns:
        model ()
    
""" 
def build_ann_model(inputs):
    # build a model
    model = Sequential()
    model.add(Dense(1000, input_shape=(inputs.shape[1],), activation='relu')) # input shape is (features,) 1000
    model.add(Dense(5000, activation='relu')) # hidden layer
    model.add(Dense(3, activation='softmax'))
    model.summary()

    # compile the model
    model.compile(optimizer='rmsprop',
                   # this is different instead of binary_crossentropy (for regular classification)
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# =============================================================================
# Textual transformers
# =============================================================================
transformers = [countvect, tfidf]

# =============================================================================
# ML classifiers
# =============================================================================
classifiers = [mnb, lr,xgb]

# =============================================================================
# Selection of best ML model: according to the training experiments
# ============================================================================= 
# ML "name"
model_name = {
    0: 'tf-idf-xgboost',
    1: 'tf-idf-xgboost',
    2: 'count-vect-xgboost'
}
def get_best_ml_name(data_index):
    if data_index in model_name.keys():
        return model_name[data_index]
    else:
        return 'tf-idf-xgboost'

# ML algorithm
best_classifiers = {
    0:xgb,
    1:xgb,
    2:xgb
}
def get_best_classifier(data_index):
    if data_index in best_classifiers.keys():
        return best_classifiers[data_index]
    else: return xgb

# NLP transformer 
best_transformer = {
    0:tfidf,
    1:tfidf,
    2:countvect
}
def get_best_transformer(data_index):
    if data_index in best_transformer.keys():
        return best_transformer[data_index]
    else: return tfidf

