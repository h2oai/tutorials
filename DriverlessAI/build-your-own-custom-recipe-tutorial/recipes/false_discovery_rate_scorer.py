"""False Discovery Rate: `FP / (FP + TP)` for binary classification""" 
#Enable base class         
import typing
import numpy as np
from h2oaicore.metrics import CustomScorer
from sklearn.preprocessing import LabelEncoder 
from sklearn.metrics import confusion_matrix
                        
class FalseDiscoveryRateScorer(CustomScorer):


    #Optimizing the scorer
    _threshold = 0.5
    _binary = True
    _regression = False
    _multiclass = False
    _maximize = False 
    _perfect_score = 0

    #Implementing Scoring Metric
    def score(self,
           actual: np.array,               
           predicted: np.array,
           sample_weight: typing.Optional[np.array] = None,
           labels: typing.Optional[np.array] = None) -> float:
 
        lb = LabelEncoder()
        labels = lb.fit_transform(labels)
        actual = lb.transform(actual)
                        
        predicted = predicted >= self.__class__._threshold # probability -> label
        cm = confusion_matrix(actual, predicted, sample_weight=sample_weight, labels=labels)
 
        tn, fp, fn, tp = cm.ravel()
        return fp / (fp + tp)