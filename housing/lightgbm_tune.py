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
from sklearn.model_selection import RepeatedKFold, KFold
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


def n_estimators_objective(trial, X, y, n_estimators=1000):
    """objective for n_estimators sample"""
    lgbm_params = {
        "objective": "regression",
        "n_estimators": n_estimators, # time 1k trials
        "learning_rate": 5e-3,
        "verbose": -1,
        # sampled params
        "num_leaves": trial.suggest_int("num_leaves", 7, 4095),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 100),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
    }

    sfold = RepeatedKFold(n_splits=5, n_repeats=2, random_state=1234)

    reg_pipe = model.get_reg_pipeline(reg_strategy="lightgbm", reg_params=lgbm_params)
    cv_results = model.cv_with_validation(
        reg_pipe,
        X,
        y,
        sfold,
        callbacks=model.common_cv_callbacks() | {"lgbm_metrics": model.lgbm_fit_metrics},
    )
    cv_results_df = _cv_results_df(cv_results)

    eval_test = cv_results_df.mean(numeric_only=True)
    for k, v in eval_test.items():
        trial.set_user_attr(k, v)
    return eval_test["test_l2"]


def n_estimators_sample(n_trials=20, outdir=".", n_estimators=1000):
    """run optuna lightgbm n_estimators samples"""
    sql_file = (
        f'sqlite:///{str(utils.WORKING_DIR / outdir / "lgbm_n_estimators_sample.db")}'
    )

    study = optuna.create_study(
        storage=sql_file,
        load_if_exists=False,
        study_name="lgbm_n_estimators",
        pruner=optuna.pruners.NopPruner(),
        direction="minimize",
        sampler=optuna.samplers.RandomSampler(),
    )

    warnings.simplefilter("ignore")  # to suppress multiple callback warning
    # pre-load data for trials
    raw_train_df, target_ds = load_prep.raw_train()
    X, y = raw_train_df, load_prep.transform_target(target_ds)
    study.optimize(
        functools.partial(n_estimators_objective, X=X, y=y, n_estimators=n_estimators),
        n_trials=n_trials
    )
    warnings.resetwarnings()
    return study


if __name__ == "__main__":
    fire.Fire()