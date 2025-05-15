import os
import csv
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as tts

path = os.path.abspath(os.getcwd())

Trials_design = {
    "data_name": 'trials_design',
    "error_types": ["inconsistencies"],
    "labels":['open', 'arms', 'double_blind', 'single_blind', 'controlled', 'parallel_group', 'crossover', 'randomised'],
    "class_counts":[2,  3,  2,  2,  2,  2, 2, 2],
    "ml_task": "classification",
    "features":["title"],
    "keys":['eudract_number', 'protocol_country_code']
}

Trials_population = {
    "data_name": 'trials_population',
    "error_types": ["inconsistencies"],
    "labels":['elderly',  'adults',  'adolescents',  'children',  'female',  'male', 'healthy_volunteers'],
    "class_counts":[2,  2,  2,  2,  2,  2, 2],
    "ml_task": "classification",
    "features":["inclusion"],
    "keys":['eudract_number', 'protocol_country_code']
}

Allergens = {
    "data_name": 'allergens',
    "error_types": ["inconsistencies"],
    "labels":['nuts', 'milk'],# 'gluten', 'soy', 'peanut', 'eggs'],
    "class_counts":[3,  3,  3,  3,  3,  3],
    "ml_task": "classification",
    "features":["ingredients"],
    "keys":['code', 'source']
}

# domain of dataset 
datasets = [Trials_design, Trials_population, Allergens]

def train_test_split(data, label):
        dtrain, dvalid = tts(data, test_size=0.1)
        iter = 0
        while len(dvalid[label].unique()) != len(dtrain[label].unique()):
            dtrain, dvalid = tts(data, test_size=0.1)
            iter += 1
            print(str(iter) + "th iteration")
        return dtrain, dvalid

class Dataset:
    def __init__(self, data_index, parker):
        self.data_index = data_index
        self.dataset = datasets[data_index]
        self.keys = self.dataset['keys']
        self.partial_key = self.dataset['keys'][0]
        self.labels = self.dataset['labels']
        self.features = self.dataset['features'][0] # the model is only using only one attribute feature (stored in list)
        self.data_name = self.dataset['data_name']
        self.data_dir = "./data/" + self.dataset["data_name"]
        self.gs_fileName =  self.data_dir + "/" + self.dataset["data_name"] + "_golden_standard.csv"
        self.data_fileName =  self.data_dir + "/" + self.dataset["data_name"]
        self.parker = parker
        self.label_encoders = {}

    def read_train_csv(self):
        """ Get the dataset representing the ground truth values
            Make sure that the domain checks are correct
        Args:
            data_name (str): refers to the name representing the dataset
            parker (Boolean): indicates whether get the dataset cleaned by Parker Engine
        Returns: gs (pandas.DataFrame) 
        """
        dtrain = self.read_data_csv("train")

        return self.get_encoded_labels(dtrain)

    def read_test_csv(self):
        return self.read_data_csv("test")

    def read_gs_csv(self):
        """ get the dataset representing the ground truth values
        Args:
            data_name (str): refers to the name representing the dataset
        Returns: gs (pandas.DataFrame) 
        """
        return self.read_data_csv("gs")

    # ===========================================================
    # Create a label encoder for each attribute subject to repair
    # ===========================================================
    def get_encoded_labels(self, data):
        """ Encode the labels
        Parameters:
            data (Dataframe): a dataset
        Returns:
            data(Dataframe): dataset having a set of columns (referred as labels) converted to numerical representations (if required)
            encoder (dict): a mapping between the original value and the numerical value per label
        """      
        if self.data_name == "trials_design":
            # 'placebo', 'active_comparator' are ned to be converted to check the selection rules constraints            
            labels = self.labels + ['placebo', 'active_comparator']
            logging.info(f"\n data columns: {list(data.columns)}")
            for label in labels :   
                classes = data[label].unique()
                n_class = len(classes)            

                # restict the classes to binary classes if Yes/No
                if 'Yes' in classes or 'No' in classes:
                    n_class = 2
                    classes = ['Yes', 'No']
                    
                self.label_encoders[label] = self.get_labelEncoder(n_class, classes, label)
                print(f"{label}-encoder: {self.label_encoders[label]}")
                data[label] = data[label].map(self.label_encoders[label])

                # Domain checks constraints   
                if label == 'arms':
                    print(data[label].unique())
                    # keep rows with only valid label values
                    data = data[(data['arms'] == self.label_encoders[label]['2+']) | (data['arms'] == self.label_encoders[label]['1'])|
                    (data['arms'] == self.label_encoders[label]['0'])]
                else: 
                    data = data[(data[label] == self.label_encoders[label]['Yes']) | (data[label] == self.label_encoders[label]['No'])]
                
        return data, self.label_encoders

    def get_labelEncoder(self, n_class, classes, label): 
        """
            1st variety of encoding
            Args:
                a: label   
                df: dataframe (column) or series of values
            Results:
                label_dict (Dict): { original value : numerical value}
        """
        # customized encoder        
        label_dict = {}
        if 'Yes' in classes and 'No' in classes:
            label_dict = {'Yes': int(1), 'No': int(0)}
            return label_dict
        
        i = n_class - 1
        for v in classes:
            print(v,i)
            label_dict[v] = int(i)
            i -= 1
        return label_dict

    def decode(self, label, y):
        """
            Map a list of values from (coonverted) "numerically" types to "originally" value type
            label (string): name of the attribute
            y (list or array or Series): list of values 
        """
        y_dec = []    
        if self.label_encoders is not None and label in self.label_encoders:
            decoder = {}
            for key, value in self.label_encoders[label].items(): 
                decoder[value] = key
            y_dec = list(map(decoder.get, y))
        return y_dec

    def encode(self, label, data):
        """
        Read if exists the encoder function for a specific attribute
        encoder (dict): mapper key (the original value) to value (encoded value)
        label (string): the attribute
        data (DataFrame)
        
        Returns: the encoder function for the attribute, the list of encoded original values and the list of encoded ground truth values
        
        """
        # get the encoder if exists
        y_orig = data[label].values
        y_gs = data[label + '_gs'].values
        if label in self.label_encoders: 
            y_orig = data[label].map(self.label_encoders[label]).values # encode orig values 
            y_gs = data[label + '_gs'].map(self.label_encoders[label]).values  #encode gs 

        return y_orig, y_gs


    # ================================================================
    # Retrieve a dataset based on its role (train, test, ground truth)
    # ================================================================
    def read_data_csv(self, data_role):
        """ get the dataset representing the ground truth values
            with removing the missing and duplicates rows
        Args:
            self (Dataset)
            dataRole (str): should be either "gs", "train", or "test"
        Returns: gs (pandas.DataFrame) 
        """

        gs = pd.read_csv(self.gs_fileName, quoting=csv.QUOTE_NONNUMERIC)
        gs = gs[[self.partial_key] + self.labels] 

        if data_role == "gs":
            return gs       
        else:
            data = pd.read_csv(self.data_fileName + "_" + data_role + ".csv", quoting=csv.QUOTE_NONNUMERIC)        
            if self.parker: 
                data = pd.read_csv(self.data_fileName + "_parker" + "_" + data_role + ".csv", sep = ",", quoting=csv.QUOTE_NONNUMERIC)
            logging.info(f'relative path {self.data_fileName} \n -- {data_role}: before removing missing and duplicates rows {data.shape}')
            
            if data_role == "train":            
                data.dropna(subset= [self.features]+[label for label in self.labels], inplace=True)
                data = data[data[self.features].str.len() > 1].copy()
                data.drop_duplicates(subset = self.keys + [self.features], keep = 'last', inplace=True, ignore_index=True)
                data = data[~(data[self.partial_key].isin(gs[self.partial_key]))]
            
            if data_role == "test":
                #data = data.merge(gs, how='inner', on=self.partial_key, suffixes=('', '_gs'))
                data = data[data[self.partial_key].isin(gs[self.partial_key])]            
        
            logging.info(f'\n -- {data_role}: after removing missing and duplicates rows {data.shape}')
            return data

    # =================================================
    # Stats about the label attribute
    # =================================================
    def get_class_stats(self, label_values):
        """
        Function to get the data imbalance ratio
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

    # # =============================================================================
    # # Remove rows from DataFrame 
    # # where there are inconsistencies between two columns
    # # =============================================================================
    # def remove_inconsistent_rows(self, data, text_column, binary_column, keyword):
    #     """
    #     Remove rows from DataFrame where there are inconsistencies between two columns.
        
    #     Parameters:
    #     - df: pandas DataFrame
    #     - text_column: str, name of the text column
    #     - binary_column: str, name of the binary column
    #     - keyword: str, keyword to check for in the text column
        
    #     Returns:
    #     - DataFrame with inconsistent rows removed
    #     """
    #     # Check if the keyword is present in the text column and binary column is not 1
    #     inconsistent_rows = data[(data[text_column].str.contains(keyword, na=False, regex=True, case=False)) \
    #                            & (data[binary_column] == 0)]
            
    #     # Remove inconsistent rows from the original DataFrame
    #     data_cleaned = df.drop(inconsistent_rows.index)
    #     print(binary_column, ': removed', len(inconsistent_rows.index))
        
    #     return data_cleaned                       
