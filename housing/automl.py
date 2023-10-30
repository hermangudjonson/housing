"""
Modelling with Autogluon and FlaML

Use budget-based model search on each platform.
Compare raw training data vs minor pre-processing.
"""

import flaml
import mlflow
import pandas as pd
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
