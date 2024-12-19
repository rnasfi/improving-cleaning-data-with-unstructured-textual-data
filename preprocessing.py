import pandas as pd
import numpy as np
import re

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import gensim
from nltk.tokenize import word_tokenize
# import spacy

import string

from sklearn import preprocessing, model_selection, metrics, linear_model, tree
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


#-------prepare the data by encoding, or removing duplicate/null values
def preprocess(dataset, data):
    features = dataset['features'][0]
    labels = dataset["labels"]
    keys = dataset["keys"]
    data, encoder = get_encoded_labels(data, dataset)
    data.dropna(subset= [features]+[label for label in labels], inplace=True)
    data = data[data[features].str.len() > 1].copy()
    data.drop_duplicates(subset = keys + [features], keep = 'last', inplace=True, ignore_index=True)
    
    return data, encoder

#--------Get the data imbalance ratio---
def get_class_stats(label_values):
    """
    Args:
        label_values: a list of a label values
    Returns:
        n_class (int): number of classes
        unique_classes (List): list of possible classes that a label can have
        class_counts (List): list of the number if each class in "label_values"
        ir: imbalance ratio
    """
    unique_classes, class_counts = np.unique(label_values, return_counts=True)
    n_class = len(unique_classes)
    ir = round(class_counts.min()/class_counts.max(), 3)
    return n_class, unique_classes, class_counts, ir
#
#---1st variety of encoding
# a: label
# df: dataframe (column) or series of values
def get_labelEncoder(n_class, classes, label):
    
    label_dict = {}
    if 'Yes' in classes and 'No' in classes:
        label_dict = {'Yes': int(1), 'No': int(0)}
        return label_dict
    
#     if n_class > 2:      
    i = n_class - 1
    for v in classes:
        label_dict[v] = int(i)
        i -= 1
    return label_dict

### 1st variety of decoding
def decode(encoder, label, y):
    y_dec = []    
    if len(encoder)>0:
        decoder = {}
        for key, value in encoder.items(): decoder[value] = key
        y_dec = list(map(decoder.get, y))
    return y_dec

### 1st variety of decoding
def encode(encoder, label, data):
    """
    Read if exists the encoder function for a specific attribute
    encoder (dict): key (the original value) and value (encoded value)
    label (string): the attribute
    data (DataFrame)
    
    Returns: the encoder function for the attribute, the list of encoded original values and the list of encoded ground truth values
    
    """
    # get the encoder if exists
    encoder_label = {}
    y_orig = data[label] 
    y_gs = data[label + '_gs']
    if len(encoder)>0:
        if len(encoder[label])>0: 
            y_orig = data[label].map(encoder[label]) # encode orig values 
            y_gs = data[label + '_gs'].map(encoder[label])  #encode gs  dtest[a].map(encoder[a]) # encode orig values 
            encoder_label = encoder[label]
    return encoder_label, y_orig, y_gs

def get_encoded_labels(df, dataset):
    """ Encode the labels
    Parameters:
        df (Dataframe): a dataset
        dataName (str): string representing the dataset
    Returns:
        df(Dataframe): dataset having a set of columns (referred as labels) converted to numerical representations (if required)
        encoder (dict): a mapping between the original value and the numerical value per label
    """
    encoder = {}
    t = []
    
    labels = dataset["labels"]
    
    if dataset["data_dir"] == "trials_design":
        # 'placebo', 'active_comparator' are ned to be converted to check the selection rules constraints
        labels = labels + ['placebo', 'active_comparator']
        for label in labels :   
            classes = df[label].unique()
            n_class = len(classes)            
            #print(label, 'classes', classes, n_class)

            # restict the classes to binary classes if Yes/No
            if 'Yes' in classes or 'No' in classes:
                n_class = 2
                classes = ['Yes', 'No']
                
            encoder[label] = get_labelEncoder(n_class, classes, label)
            df[label] = df[label].map(encoder[label])
            
    return df, encoder

##################################
## remove common words and tokenize
##################################
stoplist = set('for a of the and to in'.split()).union(set(stopwords.words('english')))

def remove_unecessary_words(text, stoplist):
    txt = ""
    for word in re.sub('[^a-zA-Z]+', ' ', text.lower()).split():
        if word not in stoplist: txt = txt + " " + word
    return txt


#     text = ''.join([re.sub('[^a-zA-Z]+', ' ', word.lower()) for word in text if word not in string.punctuation])


# Text cleaning and preprocessing
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # Keep only alphabetic characters
    text = text.lower()  # Convert to lowercase
    tokens = text.split()
    stop_words = set(stopwords.words('english')).union(set(stopwords.words('german')))
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]  # Apply stemming
    return ' '.join(tokens)# Text cleaning and preprocessing

# Indicator to tell whether 
# the label can be considered as a trace or not
def traces(text, label):
    text = remove_unecessary_words(text.strip(), stoplist).lower()
    
    if label == 'nuts':
        pres = 0
        for n in ['nut'] + ['almonds']:
            detected = closest_word_before(text, n)
#             print(n, 'detected', detected, 'prev',pres)
            
            if detected == 2:# allergen -> nuts                
                pres = 2

            if detected == 1 and pres != 2:
                pres = 1
        if pres == 1 or pres == 2: return pres
   
    trace = closest_word_before(text, label)
    return trace

def closest_word_before(text, w3):
    w1 = "allergen"
    w2 = "traces"     
    
    # Split the text into words
    words = re.sub(r'[^a-zA-Z]', ' ', text).split()
    
    # Initialize variables to store the positions of w1 and w2 relative to w3
    pos_w1 = None
    pos_w2 = None
    
    # Initialize variables to store the closest word to w3
    closest_word = None
    min_distance = float('inf')  # Initialize with positive infinity
    
    # Iterate over the words in the text
    for i, word in enumerate(words):
        # Check if the word is w1 or w2
        if word == w1:
            pos_w1 = i
        elif word == w2:
            pos_w2 = i

        # Calculate distances from w3
        if word == w3 or re.search(rf'\b{w3}(s?)\b', word.lower()):
#             print('i', i, '-traces', pos_w2, '-allergen', pos_w1, 'min_distance', min_distance)
            
            # Check if w1 or w2 appeared before w3
            if pos_w1 is not None or pos_w2 is not None:
                if pos_w1 is not None:
                    distance_w1 = i - pos_w1
                else: 
                    distance_w1 = float('inf')

                if pos_w2 is not None:
                    distance_w2 = i - pos_w2
                else: 
                    distance_w2 = float('inf')                   

                # Update the closest word
                if distance_w1 < distance_w2 and distance_w1 < min_distance:
                    closest_word = w1
                    min_distance = distance_w1
                elif distance_w2 < distance_w1 and distance_w2 < min_distance:
                    closest_word = w2
                    min_distance = distance_w2
            else:
                closest_word = w3
                break

    if closest_word == w1 or closest_word == w3:  return 2
    elif closest_word == w2:    return 1
    else: return 0
    
def correct_labels(df, labels):
    for i, r in df.iterrows():
        for l in labels:            
            if (r[f"{l}_trace"] == 1) and (r[l] == 2):
                df.loc[i,l] = 1
            if (r[f"{l}_trace"] == 2) and (r[l] == 1):
                df.loc[i,l] = 2                
            if np.isnan(r[l]) or (r[l] == 0):
                df.loc[i,l] =  r[f"{l}_trace"]
    return df


##################################
## Add the folds to the training dataset for cross validation
##################################
def folding(attribute, df, splits):
    #if len(df[attribute].unique()) < splits: splits = len(df[attribute].unique())

    print('nb folders:', splits, 'nb unique values:', len(df[attribute].unique()))	        
    df["kfold"] = -1
    df = df.sample(frac=1).reset_index(drop=True)

    kf = model_selection.StratifiedKFold(n_splits=splits)

    for f, (t_, v_) in enumerate(kf.split(X = df, y=df[attribute].values)):
        df.loc[v_, 'kfold'] = f
    
    return df, splits


##################################
# Remove rows from DataFrame where there are inconsistencies between two columns
##################################
def remove_inconsistent_rows(df, text_column, binary_column, keyword):
    """
    Remove rows from DataFrame where there are inconsistencies between two columns.
    
    Parameters:
    - df: pandas DataFrame
    - text_column: str, name of the text column
    - binary_column: str, name of the binary column
    - keyword: str, keyword to check for in the text column
    
    Returns:
    - DataFrame with inconsistent rows removed
    """
    # Check if the keyword is present in the text column and binary column is not 1
    inconsistent_rows = df[(df[text_column].str.contains(keyword, na=False, regex=True, case=False)) \
                           & (df[binary_column] == 0)]
        
    # Remove inconsistent rows from the original DataFrame
    df_cleaned = df.drop(inconsistent_rows.index)
    print(binary_column, ': removed', len(inconsistent_rows.index))
    
    return df_cleaned

#===========================================
# Functions of transforming textual features
#===========================================
def counter(texts):
    # Convert the text data into a bag-of-words representation
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)
    return X

def tdidf(texts):
    # Create the TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer()
    # Fit and transform the corpus to obtain the TF-IDF matrix
    X = tfidf_vectorizer.fit_transform(texts)
    
    # Get the feature names (words) that are used as columns in the TF-IDF matrix
    feature_names = tfidf_vectorizer.get_feature_names_out()
    print("number of Feature Names:", len(feature_names)) # = 56093
    
    return X