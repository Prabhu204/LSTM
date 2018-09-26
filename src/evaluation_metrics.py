"""
author: Prabhu

"""
import numpy as np
from sklearn import metrics

def get_metrics(true_value, predicted, list_metrics):
    predicted = np.argmax(predicted, -1)
    output= {}
    if 'Accuracy' in list_metrics:
        output['Accuracy'] = metrics.accuracy_score(true_value, predicted)
    if 'Loss' in list_metrics:
        try:
            output['Loss'] = metrics.log_loss(true_value, predicted)
        except ValueError:
            output['Loss'] = -1
    if 'Confusion_matrix' in list_metrics:
        output['Confusion_matrix'] = metrics.confusion_matrix(true_value,predicted)
    return output

