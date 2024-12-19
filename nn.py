import preprocessing as tp

import torch
from torch import nn
from torch.optim import AdamW
from torch.nn.functional import softmax
from torch.utils.data import DataLoader, Dataset#
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModel, LlamaTokenizerFast

from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
import datetime
from sklearn.metrics import accuracy_score, classification_report

## Create a custom dataset class for text classification
class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        """
        Initialize Dataset Object.    
        Parameters:
        - texts: list of texts
        - labels: attributes to be predicted from the texts
        - tokenizer: tools to tokeize the words in the texts
        - max_length: the maximum count of words in the whole set of texts
        Returns:
        - TextClassificationDataset
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        encoding = self.tokenizer(text, return_tensors='pt', add_special_tokens=True, 
                                  max_length=self.max_length, 
                                  padding='max_length', truncation=True, return_attention_mask=True)
        # sourceTensor.clone().detach().requires_grad_(True)
        return {'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(), 'label': torch.tensor(label)}


class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name) # ## Specify the pretrined model (NN) for classification
        self.linear1 = nn.Linear(768, 256)
        # During training, randomly zeroes some of the elements of the input tensor with probability p.
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.batch_size = 20

    def forward(self, input_ids, attention_mask):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask) # add two outputs
            pooled_output = outputs.pooler_output # here the start the feeding of the neural network
            x = self.dropout(pooled_output)
            logits = self.fc(x)
            return logits

    def _getTokenizer(self, bert_model_name):
        return BertTokenizer.from_pretrained(bert_model_name)

    def _train(self, data_loader, optimizer, scheduler, device):
        self.train()
        for batch in data_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = self(input_ids=input_ids, attention_mask=attention_mask)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

    def _predict(self, text, tokenizer, device, max_length=128):
        self.eval()
        encoding = tokenizer(text, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        with torch.no_grad():
            outputs = self(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)

        return preds

    def _evaluate(self, data_loader, device):
        self.eval()
        dist_proba = []
        predictions = []
        actual_labels = []
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(device) #
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device) # 
                outputs = self(input_ids=input_ids, attention_mask=attention_mask)
                probabilities = softmax(outputs, dim = 1)
                _, preds = torch.max(outputs, dim=1)
                dist_proba.extend(probabilities.cpu().tolist())
                predictions.extend(preds.cpu().tolist())
                actual_labels.extend(labels.cpu().tolist())
        return predictions, actual_labels, dist_proba

## Some variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_length = 128


def NN_train(dtrain, label, ir, unique_classes, bert_model_name, learning_rate, num_epochs, features, seed = 42):
    #build bert classifier
    model = BERTClassifier(bert_model_name, len(unique_classes)).to(device)
    tokenizer = model._getTokenizer(bert_model_name)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
        
    epochs = []
    for epoch in range(num_epochs):
        print(f'epoch: {epoch}/{num_epochs}' )
        res_epoch = {}    
        #split the train data
        df_train, df_valid = train_test_split(dtrain, test_size=0.1, random_state=seed) #
        print('train:', df_train.shape, 'valid:', df_valid.shape)
        # oversampling/imbalance ratio
        if ir < 0.3:
            over_down_sample = RandomOverSampler(sampling_strategy='not majority', random_state=seed)
            X, y = over_down_sample.fit_resample(df_train[[features]], df_train[label].values)
            print('oversampling:', ir, 'new shape',  X.shape, 'columns:', X.columns)                  
        else:
            X = df_train[[features]] 
            y = df_train[label].values
        #list of texts
        train_texts = [tp.remove_unecessary_words(row[features], tp.stoplist) for i, row in X.iterrows()]
        valid_texts = [tp.remove_unecessary_words(row[features], tp.stoplist) for i, row in df_valid.iterrows()] 
        #tensor
        labels_train = torch.tensor(y).long()    
        labels_valid = torch.tensor(df_valid[label].values).long()    
        #dataset
        train_dataset = TextClassificationDataset(train_texts, labels_train, tokenizer, max_length)
        val_dataset = TextClassificationDataset(valid_texts, labels_valid, tokenizer, max_length)          
        #dataloader
        train_dataloader = DataLoader(train_dataset, batch_size=model.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=model.batch_size)                
        total_steps = len(train_dataloader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps) 
               
        # start trainin ann model
        current_time = datetime.datetime.now()        
        model._train(train_dataloader, optimizer, scheduler, device)

        # evaluate with valid data
        pred, actual_values, proba_valid = model._evaluate(val_dataloader, device)
        res_epoch['accuracy'] = accuracy_score(pred, actual_values)      
        print(label, f"Validation Accuracy: {res_epoch['accuracy']:.4f}")
                
        previous_time = current_time
        current_time = datetime.datetime.now()
        res_epoch['duration_secs'] = (current_time - previous_time).total_seconds()
        print('done at', current_time, '')
        epochs.append(res_epoch)       
    return model, epochs


def NN_test(bert_model_name, model, dtest, label, features):
        # outputs: probability distribution	
        tokenizer = model._getTokenizer(bert_model_name)	
        labels_test = torch.tensor(dtest[f"{label}_gs"].values).long()
        test_texts = [tp.remove_unecessary_words(row[features], tp.stoplist) for i, row in dtest.iterrows()]    
        test_dataset = TextClassificationDataset(test_texts, labels_test, tokenizer, max_length)
        test_dataloader = DataLoader(test_dataset, batch_size=model.batch_size)
        # evaluate
        y_pred, y_test, probabilities = model._evaluate(test_dataloader, device)
        return y_pred, probabilities