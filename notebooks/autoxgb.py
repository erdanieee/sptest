"""
Author: Carlos Loucera <carlos.loucera@juntadeandalucia.es>
Author: Daniel Lopez <daniel.lopez.lopez@juntadeandalucia.es>

XGB optimization module.
"""

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.exceptions import NotFittedError
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier

import os
from dotenv import find_dotenv, load_dotenv

dotenv_filepath = find_dotenv()
load_dotenv(dotenv_filepath)
project_path = os.path.dirname(dotenv_filepath)

NUM_CPUS = int(os.getenv("NUM_CPUS"))


def optimize_params(X, y, params_space, validation_split=0.3, seed=42, cv=False):
    """Estimate a set of 'best' model parameters."""
    # Split X, y into train/validation
    if cv:
        if hasattr(X, "values"):
            X_train = X.values[cv[0], :]
            X_val = X.values[cv[1], :]
        else:
            X_train = X[cv[0], :]
            X_val = X[cv[1], :]
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=validation_split,
            stratify=y,
            random_state=seed
        )

    # Estimate XGB params
    def objective(_params):
        _clf = XGBClassifier(n_estimators=100,
                             max_depth=int(_params['max_depth']),
                             learning_rate=_params['learning_rate'],
                             min_child_weight=_params['min_child_weight'],
                             subsample=_params['subsample'],
                             colsample_bytree=_params['colsample_bytree'],
                             gamma=_params['gamma'],
                             silent=1,
                             verbose=False,
                             verbose_eval=False,
                             random_state=seed,
                             nthread=NUM_CPUS)
        _clf.fit(X_train, y_train,
                 eval_set=[(X_train, y_train), (X_val, y_val)],
                 eval_metric='map',
                 early_stopping_rounds=30)
        y_pred_proba = _clf.predict_proba(X_val)[:, 1]
        average_precision = average_precision_score(y_val, y_pred_proba)
        return {'loss': 1.0 - average_precision, 'status': STATUS_OK}

    trials = Trials()
    return fmin(fn=objective,
                space=params_space,
                algo=tpe.suggest,
                max_evals=100,
                trials=trials,
                verbose=0)


class OptimizedXGB(BaseEstimator, ClassifierMixin):
    """XGB with optimized parameters.

    Parameters
    ----------
    custom_params_space : dict or None
        If not None, dictionary whose keys are the XGB parameters to be
        optimized and corresponding values are 'a priori' probability
        distributions for the given parameter value. If None, a default
        parameters space is used.
    """

    def __init__(self, custom_params_space=None):
        self.custom_params_space = custom_params_space
#         self.cv = cv

    def fit(self, X, y, validation_split=0.3, seed=42):
        """Train a XGB model.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Data.

        y : ndarray, shape (n_samples,) or (n_samples, n_labels)
            Labels.

        validation_split : float (default: 0.3)
            Float between 0 and 1. Corresponds to the percentage of 
            samples in X which will be used as validation data to 
            estimate the 'best' model parameters.
        """
        # If no custom parameters space is given, use a default one.
        if self.custom_params_space is None:
            _space = self.get_default_xgb_space()
        else:
            _space = self.custom_params_space

        # Estimate best params using X, y
        opt = optimize_params(X, y, _space, validation_split, seed)

        # Instantiate `xgboost.XGBClassifier` with optimized parameters
        best = XGBClassifier(n_estimators=10000,
                             max_depth=int(opt['max_depth']),
                             learning_rate=opt['learning_rate'],
                             min_child_weight=opt['min_child_weight'],
                             subsample=opt['subsample'],
                             gamma=opt['gamma'],
                             verbose=False,
                             verbose_eval=False,
                             silent=1,
                             nthread=NUM_CPUS,
                             colsample_bytree=opt['colsample_bytree'])
        best.fit(X, y)
        self.best_estimator_ = best
        return self

    def predict(self, X):
        """Predict labels with trained XGB model.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)

        Returns
        -------
        output : ndarray, shape (n_samples,) or (n_samples, n_labels)
        """
        if not hasattr(self, 'best_estimator_'):
            raise NotFittedError('Call `fit` before `predict`.')
        else:
            return self.best_estimator_.predict(X)

    def predict_proba(self, X):
        """Predict labels probaiblities with trained XGB model.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)

        Returns
        -------
        output : ndarray, shape (n_samples,) or (n_samples, n_labels)
        """
        if not hasattr(self, 'best_estimator_'):
            raise NotFittedError('Call `fit` before `predict_proba`.')
        else:
            return self.best_estimator_.predict_proba(X)

    def get_default_xgb_space(self):
        _space = {
            'learning_rate': hp.uniform('learning_rate', 0.0001, 0.05),
            'max_depth': hp.quniform('max_depth', 8, 15, 1),
            'min_child_weight': hp.quniform('min_child_weight', 1, 5, 1),
            'subsample': hp.quniform('subsample', 0.7, 1, 0.05),
            'gamma': hp.quniform('gamma', 0.9, 1, 0.05),
            'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 0.7, 0.05)
        }
        
        return _space
