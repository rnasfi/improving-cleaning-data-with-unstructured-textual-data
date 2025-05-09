### phase 1. Data training selection
The selection of the samples to train ML model should follow these rules:
- first: NULL-value are not allowed at the cell samples.
- second: the consistency at the cell level (structured data) should be validated (e.g. regular expresion, domain restriction, ...)
- third: the consistency at the tuple sample is validated (e.g. selection rules, ...).
- fourth: the consistency at the schema samples is validated (e.g. functional dependencies, ...)


### phase 2. ML training
In ``ML_train'', you can test for three different dataset the cases of removing or keeping inconsistencies in the datasets.
There are also three different ML algorithms (Logistic regression, Multinomial Bayesian, and XGBoost), as well two different NLP method (TF-IDF and BOW)  to evaluate the repair performance.
N.B. The missing values in the training dataset are already removed.


### phase 3. Data repair
Once the ML model is trained, dirty samples can be  inputted to the model to be rapired.
Since the model is predicting the likely correct values, it can happens that the predcited values of an attribute $a$ are not very accurate.
To tackle with this challenge the likelihood of the predicted value is compared to a threshold $\tau_a$.
This can determine whther to replace the existing value of the attribute $a$ or not.


### Evaluating the repair for a selected ML model
The evaluation of the performance of the ML models uses the following metrics: 
- `Recall`: the ratio of correct repaired values compared to the number of erroneous values.
- `Precision`: the ratio of the correct repaired values  compared to the number of repaired (replaced) values.


### Graphical overveiw
![Alt text](./img/ml-clean-approach.svg "a cleaning ML-based approach")