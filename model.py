import xgboost
from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

"""Define the domain of ML model"""
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import RANSACRegressor
from xgboost import XGBClassifier

from sklearn import preprocessing, model_selection, metrics, linear_model, tree
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.pipeline import Pipeline as imb_pipeline
from sklearn.preprocessing import MinMaxScaler

seed = 42

def get_modelName(lang, alg):
    model_name = lang + '-' + alg
    return model_name.replace(' ','_')

# select which transformer 
def text_transformer(lang):
    if lang == 'tf-idf':
        tr = TfidfVectorizer(strip_accents = 'ascii', stop_words = 'english', token_pattern = r"\b[a-zA-Z]+\b")

    elif lang == 'count-vect':
        tr = CountVectorizer(strip_accents = 'ascii', stop_words = 'english', token_pattern = r"\b[a-zA-Z]+\b")

    else: tr = None

    return tr

## According to the training experiments
def get_best_ml(data_index):
    if data_index == 2:
        ml = 'count-vect-xgboost'
    else: ml = 'tf-idf-xgboost'
    return ml

########################################################################################
########################################################################################	
#
#### Here the selection are for training
def select_classifier(alg, n_class = None):
    clf = None

    # All the attributes will suppoosdly more or less have the same hyperparameters 
    if alg == 'logistic regression l1':
        clf = linear_model.LogisticRegression(class_weight='balanced', solver='liblinear', C=0.1, penalty='l1')

    if alg == 'logistic regression l2':
        clf = linear_model.LogisticRegression(class_weight={0:0.2, 1:0.4, 2:0.4}, solver='newton-cg', C=0.1, penalty='l2',
        multi_class = 'multinomial')

    if alg == 'decision tree':
        clf = tree.DecisionTreeClassifier(random_state=seed, criterion='entropy', max_depth=28)

    if alg == 'support vector machine':
        clf = svm.LinearSVC(class_weight='balanced', C=0.1, loss='squared_hinge')

    if alg == 'xgboost':
        clf = XGBClassifier(objective='multi:softmax', num_class=n_class, #train_data[a].unique()
                   learning_rate = 0.1, max_depth = 8, n_estimators = 220, eval_metric='mlogloss') 
                   # max_depth = 6?, sampling_method = 'gradient_based'

    if alg == 'multinomial bayesian':
        clf = MultinomialNB()

    if alg == 'gaussian bayesian':
        clf = GaussianNB()		

    if alg == 'random forest':
        clf = RandomForestClassifier(random_state=seed)
       
    if alg == 'svc':
        clf = imb_pipeline([('scaler', MinMaxScaler()), ('classify', SVC())]) 
        
    return clf

#
#### Here the selection are for tuning the hyperparameters
def select_classifier_grid(alg, n_class = None):
    #Grid Search Model
    if alg == 'decision tree':
        clf = tree.DecisionTreeClassifier(random_state=42)
        
        param_grid = {"classify__criterion": ['entropy', 'gini'], 
        "classify__max_depth": [10, 20, 28]} #2,4,6,8,12, 14, 16, 18, 22, 24, 26, 30
        return clf, param_grid
    
    if alg == 'logistic regression l1':
        clf = linear_model.LogisticRegression(class_weight='balanced')
        
        param_grid = {"classify__C":[0.01, 0.1, 1.0], "classify__penalty":["l1"],
        "classify__solver": ["liblinear", "saga"], "classify__max_iter":[1000, 5000]}
        return clf, param_grid

    if alg == 'logistic regression l2':
        clf = linear_model.LogisticRegression(class_weight='balanced', max_iter=1000)
        
        param_grid = {"classify__C":[0.01, 0.1, 1.0], "classify__penalty":["l2"],
        "classify__solver": ["newton-cg", "lbfgs", "sag", "saga"]} # , "elasticnet" Solver newton-cg supports only 'l2' or 'none' penalties
        return clf, param_grid    
    
    if alg == 'support vector machine':
        clf = LinearSVC(class_weight="balanced", random_state=42)
        
        param_grid = {"classify__loss": ['hinge', 'squared_hinge'],
        "classify__C":[ 0.00001, 0.0001, 0.001, 0.01, 0.1, 1], "classify__penalty":["l2"]} 
        return clf, param_grid
    
    if alg == 'xgboost binary':
        clf = xgboost.XGBClassifier(objective='binary:logistic')

        param_grid = {"classify__learning_rate": [0.1, 0.2], #0.01, 
        "classify__max_depth": range(6, 10, 2),
                     "classify__n_estimators": [100, 220]} # , "classify__gamma": [i/10.0 for i in range(3)]
        return clf, param_grid
    
    if alg == 'xgboost':
        clf = xgboost.XGBClassifier(objective='multi:softmax', num_class=n_class)

        param_grid = {"classify__learning_rate": [0.1, 0.2], #0.01, 
        "classify__max_depth": range(6, 10, 2),
                     #"classify__n_estimators": [50, 100, 220], 
					# "classify__sampling_method": ['gradient_based','uniform'],
					 "classify__eval_metric":['merror', 'mlogloss']}#, "classify__gamma": [i/10.0 for i in range(3)]
        return clf, param_grid  
        
    else:
        return None
    
def build_ann_model(X):
    # build a model
    model = Sequential()
    model.add(Dense(1000, input_shape=(X.shape[1],), activation='relu')) # input shape is (features,) 1000
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
# ML models parameters
# =============================================================================

countvect = {
    "name":"count-vect",
    "fn": CountVectorizer,
    "fixed_params": {}, #"strip_accents":'ascii', "stop_words":'english', "token_pattern": r"\b[a-zA-Z]+\b"
}

tfidf = {
    "name":"tf-idf",
    "fn": TfidfVectorizer,
    "fixed_params": {"strip_accents":'ascii', "stop_words":'english', "token_pattern": r"\b[a-zA-Z]+\b"},
}

mnb = {
    "name": "mnb",
    "fn": MultinomialNB,
    "fixed_params": {"alpha":0.8, "fit_prior":True, "force_alpha":True},#
    "type": "classification",
#     "hyperparams": "clf__alpha" ,
#     "hyperparams_type": "real",
    #"hyperparams_range": [0.05, 0.09]
}    
    
lr = {
    "name": "logistic_regression",
    "fn": LogisticRegression,
    "fixed_params": {"solver":"lbfgs", "max_iter":5000, "multi_class":'auto'},
    "parallelable": True,
    "type": "classification",
    "hyperparams": ["clf__C"] ,
    "hyperparams_type": ["real"],
    "hyperparams_range": [[-5, 5]]
}    
xgb = {
    "name":"xgboost",
    "fn": XGBClassifier,
    "fixed_params": {}, #"clf__learning_rate": 0.1
    "type": "classification",
    "hyperparams": ["clf__max_depth", "clf__n_estimators", "clf__objective", "clf__learning_rate"],
    "hyperparams_type": ["int", "int", "string", "real"],
    "hyperparams_range": [[6, 8, 10],[200, 300, 220], None, [.1,.2]]
}

# textual transformer
transformers = [countvect, tfidf]
# classifiers domain
classifiers = [mnb, lr,xgb]