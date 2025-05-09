import numpy as np 

def get_avg_probab(dist_proba):
    max_probas = []
    for i, proba in enumerate(dist_proba):
        max_probas.append(np.max(proba))

    avg_proba = sum(max_probas) / len(max_probas)
    mini_proba = min(max_probas)
    maxi_proba = max(max_probas)
    return avg_proba, max_probas, maxi_proba, mini_proba
   
    
def assign_repair(dist_proba, y_orig, y_pred, th):
    """
    dist_proba (List): 2D list of probability distribution of for each label's possible classes
    y_orig (List): list of label's real values
    th (float): threshold of certainty 
    Retruns (List[float]):
    list of label's repaired values
    """
    final_predictions = []
    for i, proba in enumerate(dist_proba):
        if np.max(proba) >= th  or np.isnan(y_orig[i]):
            final_predictions.append(y_pred[i])
        else: final_predictions.append(y_orig[i])
    return final_predictions

def get_precision(dist_proba, y_test, y_pred, th):
    repairs = 0
    correct_repairs = 0
    precision = 0    
    for i, proba in enumerate(dist_proba):
        if np.max(proba) >= th  or np.isnan(y_test[i]):
            repairs += 1    
            if y_pred[i] == y_test[i]: correct_repairs += 1
    if repairs != 0: precision = round(correct_repairs/repairs, 4)


def get_metrics(y_pred, y_orig, y_gs):
    """
    y_pred (List): list of label's predicted values
    y_orig (List): list of label's real values
    y_gs (List): list of label's ground truth values 
    Retruns (List[float]):
    prediction metrics: precision, recall and F-1 score
    """
    repairs = 0
    errors = 0
    correct_repairs = 0
    precision = 0
    recall = 0
    f1 = 0
    
    for i in range(len(y_pred)):
        if y_pred[i] != y_orig[i]: repairs += 1
        if y_orig[i] != y_gs[i]: errors += 1
        if y_pred[i] != y_orig[i] and y_pred[i] == y_gs[i]: correct_repairs += 1
    if repairs != 0: precision = round(correct_repairs/repairs, 2)
    if errors != 0: recall = round(correct_repairs/errors, 2)
    if precision != 0 or recall != 0: f1 = round(2 * (precision * recall)/(precision + recall), 2)
    print('correct repair', correct_repairs, 'repairs', repairs, 'errors', errors)
    return precision, recall, f1

def get_stats(y_pred, y_orig, y_gs):
    """ Provide some statistics about the repairs per cells
    
    y_pred (List): list of label's predicted values
    y_orig (List): list of label's real values
    y_gs (List): list of label's ground truth values

    Retruns (List[int]):
    prediction statistics: number of correct repairs, number of repairs and number of erroneous cells
    """    
    repairs = 0
    errors = 0
    correct_repairs = 0    
    
    for i in range(len(y_pred)):
        if y_pred[i] != y_orig[i]: repairs += 1
        if y_orig[i] != y_gs[i]: 
            errors += 1
        if y_pred[i] != y_orig[i] and y_pred[i] == y_gs[i]: correct_repairs += 1
        #if y_pred[i] == y_gs[i]: print(i)
	
    return correct_repairs, repairs, errors
    
def find_best_index(list1):
    """ get the index where it is the highest value
    list1 (List): a list of numerics
    """
    m = 0 #max(list)
    b = 0
    for l in range(len(list1)):
        if abs(m-list1[l]) > 0.01:
            m = list1[l]
            b = l
    return b

def get_all_stats(data, labels):
    """ Count for a set of attributes to be repaired, 
        the amount of errors, repairs and correct repairs
        
        data(Pandas.Dataframe): the repaired data 
                                (should include for each attribute to be repaired
                                    the original value, and the ground truth of it)
        labels (List): list of attributes to be repaired
        
        Returns(List([int]):
        prediction statistics: number of correct repairs, number of repairs and number of erroneous cells
    """
    repairs = 0
    errors = 0
    correct_repairs = 0
    
    for label in labels:
        y_pred = data[label].values
        y_orig = data[label + '_orig'].values
        y_gs = data[label + '_gs'].values
        new_correct_repairs, new_repairs, new_errors =  get_stats(y_pred, y_orig, y_gs)
        correct_repairs += new_correct_repairs
        repairs += new_repairs
        errors += new_errors
    return correct_repairs, repairs, errors
 
    
    
#########################################################################################################################    
############################# Estimate the threshold for the repairs ####################################################    
def expected_calibration_error(y_true, y_prob, v_class, n_bins=10):
    """Calculate Expected Calibration Error (ECE)"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    total_confidence = 0.0
    total_samples = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find indices of predictions that fall into the current bin
        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
        
        prop_in_bin = np.sum(in_bin) # total nb of samples in this bin      
        
        if prop_in_bin > 0:
            # Calculate accuracy and confidence for the bin
            accuracy_in_bin = np.mean(y_true[in_bin] == v_class)
            #print(bin_lower, bin_upper, 'accuracy_in_bin', accuracy_in_bin)
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            total_confidence += avg_confidence_in_bin * prop_in_bin
            #print('avg_confidence_in_bin', avg_confidence_in_bin, 'nb of samples', prop_in_bin, round(len(y_prob[in_bin])/len(y_true),2))
            #print('confidence', avg_confidence_in_bin * prop_in_bin)
            total_samples += prop_in_bin
            # Weight by the proportion of samples in the bin
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece, total_confidence/total_samples if total_samples > 0 else 0.0