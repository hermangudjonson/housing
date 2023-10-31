"""
Modelling with Autogluon and FlaML

Use budget-based model search on each platform.
Compare raw training data vs minor pre-processing.
"""

import fire
import flaml
import mlflow
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold

from housing import load_prep, model, utils


def _cv_results_df(cv_results: dict):
    """collate eval test results from cv_with_validation return"""
    folds = cv_results["eval_test"].keys()
    cv_results_df = pd.concat(
        cv_results["eval_test"].values(), keys=folds, ignore_index=True
    ).infer_objects()
    return cv_results_df


def fit_flaml(preprocess=False, time_budget=1, refit_time_budget=1, outdir=None):
    automl_params = {
        "time_budget": time_budget,  # in seconds
        "early_stop": True,
        "metric": "mse",
        "task": "regression",
        "ensemble": True,
        "log_file_name": str(utils.WORKING_DIR / outdir / "housing_flaml.log")
        if outdir
        else "",
    }

    if preprocess:
        flaml_reg = model.get_reg_pipeline(
            reg_strategy="flaml", reg_params=automl_params, as_category=True
        )
    else:
        flaml_reg = flaml.AutoML(**automl_params)

    raw_train_df, target_ds = load_prep.raw_train()
    X, y = raw_train_df, load_prep.transform_target(target_ds)

    sfold = KFold(n_splits=5, shuffle=True, random_state=1234)

    flaml_reg.fit(X, y)

    fitted_params = automl_params | {
        "time_budget": refit_time_budget,  # in seconds
        "log_file_name": "",
        "starting_points": flaml_reg[-1].best_config_per_estimator
        if preprocess
        else flaml_reg.best_config_per_estimator,
    }

    # cv results evaluation
    if preprocess:
        flaml_best_model = model.get_reg_pipeline(
            reg_strategy="flaml",
            reg_params=fitted_params,
            as_category=True,
        )
    else:
        flaml_best_model = flaml.AutoML(**fitted_params)

    cv_results = model.cv_with_validation(
        flaml_best_model,
        X,
        y,
        sfold,
        callbacks=model.common_cv_callbacks(),
    )
    cv_results_df = _cv_results_df(cv_results)

    # make submission predictions
    X_test = load_prep.raw_test()
    flaml_predict = pd.Series(
        load_prep.inv_transform_target(flaml_reg.predict(X_test)),
        name="SalePrice",
        index=X_test.index,
    )

    if outdir is not None:
        # save flaml model
        mlflow.sklearn.save_model(flaml_reg, utils.WORKING_DIR / outdir / "flaml_model")
        cv_results_df.to_csv(utils.WORKING_DIR / outdir / "flaml_best_eval_test.csv")

        # save submission predictions
        flaml_predict.to_csv("flaml_predict.csv")

    return flaml_reg


class AGProxy(BaseEstimator, RegressorMixin):
    def __init__(self, **params):
        self.params = params

    def get_params(self, deep=True):
        return self.params

    def set_params(self, **params):
        self.params.update(params)

    def fit(self, X, y):
        tab_data = TabularDataset(pd.concat([X, pd.DataFrame(y)], axis=1))

        self.params.update({"label": y.name})
        # partition between model and fit params
        model_params = self.params.copy()
        fit_keys = {"time_limit", "presets"}
        fit_params = {
            k: model_params.pop(k) for k in self.params.keys() if k in fit_keys
        }
        self.estimator_ = TabularPredictor(**model_params)

        self.estimator_.fit(tab_data, **fit_params)
        return self

    def __getattr__(self, name):
        """dispatch other methods to estimator"""
        return getattr(self.estimator_, name)


def fit_autogluon(preprocess=False, time_limit=1, presets="best_quality", outdir=None):
    ag_params = {
        "problem_type": "regression",
        "eval_metric": "root_mean_squared_error",
        "path": str(utils.WORKING_DIR / outdir / "ag_fit_output")
        if outdir is not None
        else None,
        "time_limit": time_limit,
        "presets": presets,
    }

    if preprocess:
        ag_reg = model.get_reg_pipeline(
            reg_strategy="autogluon", reg_params=ag_params, as_category=True
        )
    else:
        ag_reg = AGProxy(**ag_params)

    raw_train_df, target_ds = load_prep.raw_train()
    X, y = raw_train_df, load_prep.transform_target(target_ds)

    ag_reg.fit(X, y)
    ag_summary = (ag_reg[-1] if preprocess else ag_reg).fit_summary()

    # make submission predictions
    X_test = load_prep.raw_test()
    ag_predict = pd.Series(
        load_prep.inv_transform_target(ag_reg.predict(X_test)),
        name="SalePrice",
        index=X_test.index,
    )

    if outdir is not None:
        # save flaml model
        mlflow.sklearn.save_model(ag_reg, utils.WORKING_DIR / outdir / "ag_model")
        ag_summary["leaderboard"].to_csv(
            utils.WORKING_DIR / outdir / "ag_leaderboard.csv"
        )

        # save submission predictions
        ag_predict.to_csv("flaml_predict.csv")

    return ag_reg


if __name__ == "__main__":
    fire.Fire()
