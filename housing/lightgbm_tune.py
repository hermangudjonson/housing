"""
LightGBM hyperparameter optimization with optuna

Define optuna objective, study and runs.
Additionally compare GPU fitting performance.

tuning progression:
 - n_estimators run time
 - learning_rate sweep to evalute early stopping points
 - broad parameter sweep
"""

import functools
import warnings

import cloudpickle
import fire
import lightgbm as lgbm
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import KFold, RepeatedKFold

from housing import load_prep, model, utils


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


def n_estimators_objective(trial, X, y, n_estimators=1000, device="cpu"):
    """objective for n_estimators sample"""
    lgbm_params = {
        "objective": "regression",
        "n_estimators": n_estimators,  # time 1k trials
        "learning_rate": 5e-3,
        "verbose": -1,
        # gpu settings
        "device": device,
        "max_bin": 63 if device == "gpu" else 255,
        "gpu_platform_id": 0,
        "gpu_device_id": 0,
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

    reg_pipe = model.get_reg_pipeline(
        reg_strategy="lightgbm", reg_params=lgbm_params, as_category=True
    )
    cv_results = model.cv_with_validation(
        reg_pipe,
        X,
        y,
        sfold,
        callbacks=model.common_cv_callbacks()
        | {"lgbm_metrics": model.lgbm_fit_metrics},
    )
    cv_results_df = _cv_results_df(cv_results)

    eval_test = cv_results_df.mean(numeric_only=True)
    for k, v in eval_test.items():
        trial.set_user_attr(k, v)
    return eval_test["test_l2"]


def n_estimators_sample(n_trials=20, outdir=".", n_estimators=1000, device="cpu"):
    """run optuna lightgbm n_estimators samples"""
    sql_file = f'sqlite:///{str(utils.WORKING_DIR / outdir / f"lgbm_n_estimators_sample_{device}.db")}'

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
        functools.partial(
            n_estimators_objective, X=X, y=y, n_estimators=n_estimators, device=device
        ),
        n_trials=n_trials,
    )
    warnings.resetwarnings()
    return study


def early_stopping_objective(trial, X, y, device="cpu"):
    """objective for early stopping grid"""
    lgbm_params = {
        "objective": "regression",
        "n_estimators": 10_000,
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1.0, log=True),
        "callbacks": [lgbm.early_stopping(20, first_metric_only=True)],
        "verbose": -1,
        # gpu settings
        "device": device,
        "max_bin": 63 if device == "gpu" else 255,
        "gpu_platform_id": 0,
        "gpu_device_id": 0,
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

    reg_pipe = model.get_reg_pipeline(
        reg_strategy="lightgbm", reg_params=lgbm_params, as_category=True
    )
    cv_results = model.cv_with_validation(
        reg_pipe,
        X,
        y,
        sfold,
        callbacks=model.common_cv_callbacks()
        | {"lgbm_metrics": model.lgbm_fit_metrics},
    )
    cv_results_df = _cv_results_df(cv_results)

    eval_test = cv_results_df.mean(numeric_only=True)
    for k, v in eval_test.items():
        trial.set_user_attr(k, v)
    return eval_test["test_l2"]


def early_stopping_sweep(n_trials=20, outdir=".", device="cpu"):
    """run optuna lightgbm early stopping sweep"""
    sql_file = f'sqlite:///{str(utils.WORKING_DIR / outdir / f"lgbm_early_stopping_sweep_{device}.db")}'

    study = optuna.create_study(
        storage=sql_file,
        load_if_exists=False,
        study_name="lgbm_early_stopping",
        pruner=optuna.pruners.NopPruner(),
        direction="minimize",
        sampler=optuna.samplers.RandomSampler(),
    )

    warnings.simplefilter("ignore")  # to suppress multiple callback warning
    # pre-load data for trials
    raw_train_df, target_ds = load_prep.raw_train()
    X, y = raw_train_df, load_prep.transform_target(target_ds)
    study.optimize(
        functools.partial(early_stopping_objective, X=X, y=y, device=device),
        n_trials=n_trials,
    )
    warnings.resetwarnings()
    return study


def broad_objective(trial, X, y, device="cpu"):
    """objective for broad hpo"""
    lgbm_params = {
        "objective": "regression",
        "n_estimators": 2_000,
        "learning_rate": 5e-2,
        "callbacks": [
            optuna.integration.LightGBMPruningCallback(
                trial, "l2", valid_name="validation"
            ),
            lgbm.early_stopping(20, first_metric_only=True)
        ],
        "verbose": -1,
        # gpu settings
        "device": device,
        "max_bin": 63 if device == "gpu" else 255,
        "gpu_platform_id": 0,
        "gpu_device_id": 0,
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

    reg_pipe = model.get_reg_pipeline(
        reg_strategy="lightgbm", reg_params=lgbm_params, as_category=True
    )
    cv_results = model.cv_with_validation(
        reg_pipe,
        X,
        y,
        sfold,
        callbacks=model.common_cv_callbacks()
        | {"lgbm_metrics": model.lgbm_fit_metrics},
    )
    cv_results_df = _cv_results_df(cv_results)

    eval_test = cv_results_df.mean(numeric_only=True)
    for k, v in eval_test.items():
        trial.set_user_attr(k, v)
    return eval_test["test_l2"]


def broad_hpo(n_trials=100, timeout=3600, outdir=".", device="cpu", prune=True):
    """run optuna lightgbm broad hpo"""
    if prune:
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=200, interval_steps=50, n_min_trials=5
        )
    else:
        pruner = optuna.pruners.NopPruner()

    sql_file = f'sqlite:///{str(utils.WORKING_DIR / outdir / f"lgbm_broad_hpo_{device}.db")}'

    study = optuna.create_study(
        storage=sql_file,
        load_if_exists=False,
        study_name="lgbm_broad_hpo",
        pruner=pruner,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(),
    )

    warnings.simplefilter("ignore")  # to suppress multiple callback warning
    # pre-load data for trials
    raw_train_df, target_ds = load_prep.raw_train()
    X, y = raw_train_df, load_prep.transform_target(target_ds)
    study.optimize(
        functools.partial(broad_objective, X=X, y=y, device=device),
        n_trials=n_trials,
        timeout=timeout
    )
    warnings.resetwarnings()
    return study


if __name__ == "__main__":
    fire.Fire()
