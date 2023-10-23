"""
LightGBM hyperparameter optimization with optuna

Define optuna objective, study and runs.
Additionally compare GPU fitting performance.

tuning progression:
 - n_estimators run time
 - learning_rate sweep to evalute early stopping points
 - broad parameter sweep
"""

from housing import load_prep, model, utils

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
import optuna
import cloudpickle

import functools
import warnings
import fire


def _cv_results_df(cv_results: dict):
    """collate eval test results from cv_with_validation return"""
    folds = cv_results["eval_test"].keys()
    cv_results_df = pd.concat(
        [
            pd.concat(
                cv_results["lgbm_metrics"].values(), keys=folds, ignore_index=True
            ),
            pd.concat(cv_results["eval_test"].values(), keys=folds, ignore_index=True),
        ],
        axis=1,
    ).infer_objects()
    return cv_results_df


def n_estimators_objective(trial, X, y):
    """objective for n_estimators grid"""
    ctb_params = {
        "objective": "regression",
        "n_estimators": 1000, # time 1k trials
        "learning_rate": 5e-3,
        "allow_writing_files": False,
    }

    sfold = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=1234)

    clf_pipe = model.clf_pipeline(clf_strategy="catboost", clf_params=ctb_params)
    cv_results = model.cv_with_validation(
        clf_pipe,
        X,
        y,
        sfold,
        callbacks=model.common_cv_callbacks() | {"ctb_metrics": model.ctb_fit_metrics},
    )
    cv_results_df = _cv_results_df(cv_results)

    eval_test = cv_results_df.mean(numeric_only=True)
    for k, v in eval_test.items():
        trial.set_user_attr(k, v)
    return eval_test["test_Logloss"]


if __name__ == "__main__":
    fire.Fire()
