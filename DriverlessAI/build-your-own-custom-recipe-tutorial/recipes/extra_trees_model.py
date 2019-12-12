"""Extremely Randomized Trees (ExtraTrees) model from sklearn"""
#Extend base class
import datatable as dt
import numpy as np
from h2oaicore.models import CustomModel
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.preprocessing import LabelEncoder

class ExtraTreesModel(CustomModel):

    #Fitting the model
    def fit(self, X, y, sample_weight=None, eval_set=None, sample_weight_eval_set=None, **kwargs):

        orig_cols = list(X.names)

        if self.num_classes >= 2:
            lb = LabelEncoder()
            lb.fit(self.labels)
            y = lb.transform(y)
            model = ExtraTreesClassifier(**self.params)
        else:
            model = ExtraTreesRegressor(**self.params)

        #Can your model handle missing values??
        X = X.to_numpy() 

        model.fit(X, y)

        importances = np.array(model.feature_importances_)

        #Set model parameters
        self.set_model_properties(model=model, 
                        features=orig_cols,
                        importances=importances.tolist(), 
                        iterations=self.params['n_estimators'])

    #Get predictions
    def predict(self, X, **kwargs):

        model, _, _, _ = self.get_model_properties()

        #Can your model handle missing values??
        X = X.to_numpy()

        if self.num_classes == 1: 
            preds = model.predict(X)
        else:
            preds = model.predict_proba(X)
        return preds