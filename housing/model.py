"""
Routines for housing model generation and evaluation.
"""

from functools import partial
from time import time

import lightgbm as lgbm
import numpy as np
import pandas as pd
import sklearn.metrics as skm
from sklearn.base import BaseEstimator, RegressorMixin, clone, is_classifier
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.decomposition import PCA
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge, ElasticNetCV
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import check_cv
from sklearn.naive_bayes import CategoricalNB
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline, make_union
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
)
from sklearn.utils.metaestimators import _safe_split

from housing import load_prep

try:
    import flaml
    from housing.automl import AGProxy
except ImportError:
    flaml = None
    AGProxy = None


def get_imputer(impute_categoricals=True):
    """Pipeline step to impute scaled and filtered data.

    Data is ordinal encoded with missing indicators and scaled numeric after thsi step.

    Parameters
    ----------
    impute_categoricals : bool, optional
        whether to impute categorical features or just encode missing, by default True

    Returns
    -------
    Pipeline
    """
    if impute_categoricals:
        # ordinal encoding followed by categorical naive bayes impute
        # unknown categories are marked missing and imputed (catNB can't handle -1)
        cat_impute = make_pipeline(
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan),
            IterativeImputer(
                CategoricalNB(min_categories=25),
                initial_strategy="most_frequent",
                add_indicator=True,
                skip_complete=True,
            ),
        ).set_output(transform="pandas")
    else:
        # ordinal encoding with unknown/missing encoded as -1
        cat_impute = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
        )

    # cat_step contains names of original categorical and numeric column names
    cat_step = make_column_transformer(
        (cat_impute, make_column_selector(dtype_include="category")),
        ("passthrough", make_column_selector(dtype_exclude="category")),
        verbose_feature_names_out=False,
    )

    num_step = make_union(
        # pass ordinal encoded categoricals
        make_column_transformer(
            ("passthrough", lambda X: cat_step.transformers_[0][2]),
            remainder="drop",
            verbose_feature_names_out=False,
        ),
        # impute numeric and pass missing indicators
        make_pipeline(
            make_column_transformer(
                (
                    OneHotEncoder(
                        handle_unknown="infrequent_if_exist",
                        min_frequency=5,
                        sparse_output=False,
                    ),
                    lambda X: cat_step.transformers_[0][2],
                ),
                remainder="passthrough",
                verbose_feature_names_out=False,
            ),
            IterativeImputer(BayesianRidge(), skip_complete=True, add_indicator=True),
            make_column_transformer(
                ("passthrough", lambda X: cat_step.transformers_[1][2]),
                ("passthrough", make_column_selector(pattern="missingindicator*")),
                remainder="drop",
                verbose_feature_names_out=False,
            ),
        ),
    )
    # removes prefix append by feature union
    fix_col_names = FunctionTransformer(
        lambda X: X.set_axis(
            list(pd.Series(X.columns).str.split("__", expand=True)[1]), axis=1
        )
    )

    return make_pipeline(cat_step, num_step, fix_col_names).set_output(
        transform="pandas"
    )


def _get_ordinal_selector(impute_step):
    return lambda X: impute_step[0].transformers_[0][2]


def _get_numeric_selector(impute_step):
    return lambda X: impute_step[0].transformers_[1][2]


def get_pca_step(
    ordinal_selector,
    numeric_selector,
    passthrough=True,
    cat_n_components=5,
    num_n_components=5,
):
    """Apply PCA transform to one-hot categoricals and numeric features.

    Input is ordinal imputed data. Relies on selector funcs to identify
    remaining ordinal columns and numeric columns. PCA applied separately
    to categoricals and numeric and concatenated

    Parameters
    ----------
    ordinal_selector : Callable[[X], columns]
    numeric_selector : Callable[[X], columns]
    passthrough : bool, optional
        whether to concat input data to PCA output, by default True

    Returns
    -------
    Pipeline
    """
    cat_pca = make_pipeline(
        make_column_transformer(
            (OneHotEncoder(sparse_output=False), ordinal_selector),
            ("drop", numeric_selector),
            remainder="passthrough",
            verbose_feature_names_out=False,
        ),
        PCA(n_components=cat_n_components),
    ).set_output(transform="pandas")

    num_pca = make_pipeline(
        make_column_transformer(
            ("passthrough", numeric_selector),
            remainder="drop",
            verbose_feature_names_out=False,
        ),
        PCA(n_components=num_n_components),
    ).set_output(transform="pandas")

    # removes prefix append by feature union
    # slight trick to only remove prefix from passthrough data
    fix_col_names = FunctionTransformer(
        lambda X: X.set_axis(
            list(pd.Series(X.columns).str.removeprefix("orig__")), axis=1
        )
    )

    union_steps = [("orig", "passthrough")] if passthrough else []
    union_steps += [("cat", cat_pca), ("num", num_pca)]
    return make_pipeline(FeatureUnion(union_steps), fix_col_names).set_output(
        transform="pandas"
    )


def get_pca_pipeline(cat_n_components=5, num_n_components=5):
    # raw to PCs
    # near zero var, scale, impute, PCA
    impute_step = get_imputer()

    pca_pipe = make_pipeline(
        load_prep.preprocess_pipe(),
        impute_step,
        get_pca_step(
            _get_ordinal_selector(impute_step),
            _get_numeric_selector(impute_step),
            passthrough=False,
            cat_n_components=cat_n_components,
            num_n_components=num_n_components,
        ),
    )
    return pca_pipe


def get_gower_dist(X):
    # near zero var, scale, impute
    # impute to one hot cats to dice
    # impute to numeric to range normalized manhattan
    impute_step = get_imputer()

    impute_pipe = make_pipeline(load_prep.preprocess_pipe(), impute_step)

    # one hot encode remaining ordinal features
    impute_to_onehot = make_column_transformer(
        (OneHotEncoder(sparse_output=False), _get_ordinal_selector(impute_step)),
        ("drop", _get_numeric_selector(impute_step)),
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")
    # scale numeric to [0, 1]
    impute_to_num = make_column_transformer(
        (MinMaxScaler(), _get_numeric_selector(impute_step)),
        remainder="drop",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")

    X_imputed = impute_pipe.fit_transform(X)
    X_onehot = impute_to_onehot.fit_transform(X_imputed)
    X_num = impute_to_num.fit_transform(X_imputed)

    dice_dist = pairwise_distances(X_onehot.to_numpy(), metric="dice")
    man_dist = pairwise_distances(X_num, metric="cityblock") / X_num.shape[1]

    return (dice_dist + man_dist) / 2.0


class LGBMProxy(BaseEstimator, RegressorMixin):
    """LightGBM wrapper that conforms to sklearn interface

    specifically move callbacks and validation data to initialization
    rather than needing to be passed during the call to fit itself.
    Note catboost tries to JSON serialize all parameters (fails for dataframe)

    Parameters
    ----------
    callbacks : list, optional
        lightgbm fit callback functions
    validation : tuple (X, y), optional
        validation data, required to use early stopping
    **params : optional
        parameters to be passed to LGBMRegressor initialization
    """

    def __init__(self, callbacks=None, validation=None, **params):
        self.callbacks = callbacks
        self.validation = validation
        # lightgbm classifier
        self.estimator_ = lgbm.LGBMRegressor(**params)

    def get_params(self, deep=True):
        return self.estimator_.get_params(deep) | {
            "callbacks": self.callbacks,
            "validation": self.validation,
        }

    def set_params(self, **params):
        if "callbacks" in params:
            self.callbacks = params.pop("callbacks")
        if "validation" in params:
            self.validation = params.pop("validation")
        self.estimator_.set_params(**params)

    def fit(self, X, y):
        if self.validation is not None:
            eval_set = [self.validation, (X, y)]
            eval_names = ["validation", "training"]
        else:
            eval_set = None
            eval_names = None
        self.estimator_.fit(
            X,
            y,
            eval_set=eval_set,
            eval_names=eval_names,
            eval_metric=None,  # defaulting to training objective
            callbacks=self.callbacks,
        )
        return self

    def __getattr__(self, name):
        """dispatch other methods to estimator"""
        return getattr(self.estimator_, name)


def get_regressor(strategy="passthrough", params=None):
    params = params if params is not None else {}

    if strategy == "passthrough":
        return "passthrough"
    elif strategy == "linear":
        enet_params = {"l1_ratio": 0.9}
        params = enet_params | params
        return ElasticNetCV(**params)
    elif strategy == "lightgbm":
        return LGBMProxy(**params)
    elif strategy == "flaml":
        return flaml.AutoML(**params)
    elif strategy == 'autogluon':
        return AGProxy(**params)
    elif strategy == "custom":
        return params['estimator']
    else:
        raise ValueError(f"unimplemented strategy {strategy}")


def get_reg_pipeline(
    reg_strategy="passthrough",
    reg_params=None,
    onehot_encode=False,
    include_pca=True,
    as_category=False,
):
    reg_params = reg_params if reg_params is not None else {}

    # preprocess, impute, (optional) onehot, regressor
    impute_step = get_imputer()

    # PCA step
    pca_step = get_pca_step(
        _get_ordinal_selector(impute_step),
        _get_numeric_selector(impute_step),
        passthrough=True,
    )

    # one hot encode remaining ordinal features
    impute_to_onehot = make_column_transformer(
        (OneHotEncoder(sparse_output=False), _get_ordinal_selector(impute_step)),
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")

    # cast features to categorical before input to regressor
    cast_categoricals = make_column_transformer(
        ("passthrough", _get_numeric_selector(impute_step)),
        ("passthrough", make_column_selector(".*pca.*")),
        remainder=FunctionTransformer(lambda X: X.astype("category")),
        verbose_feature_names_out=False
    ).set_output(transform="pandas")

    reg_pipe = Pipeline(
        [
            ("preprocess", load_prep.preprocess_pipe()),
            ("impute", impute_step),
            ("PCA", pca_step if include_pca else "passthrough"),
            ("onehot", impute_to_onehot if onehot_encode else "passthrough"),
            ("categoricals", cast_categoricals if as_category else "passthrough"),
            ("regressor", get_regressor(strategy=reg_strategy, params=reg_params)),
        ]
    )
    # since we are passing info between steps, we override clone
    reg_pipe.__sklearn_clone__ = lambda: get_reg_pipeline(
        reg_strategy, reg_params, onehot_encode, include_pca, as_category
    )
    return reg_pipe


def fit_with_validation(
    reg_pipe: Pipeline,
    X_train,
    y_train,
    X_valid=None,
    y_valid=None,
):
    """fit routine for regression pipeline.

    fits first stages first in order to pass transformed
    validation data to final classifier if necessary

    Parameters
    ----------
    reg_pipe : Pipeline
        multi-step pipeline with final step regressor
    X_train : 2D array_like
    y_train : 1D array_like
    X_valid : 2D array_like, optional
        validation data to be transformed and passed to regressor, by default None
    y_valid : 1D array_like, optional
        by default None

    Returns
    -------
    reg_pipe : Pipeline, fitted
    """
    if X_valid is None:
        # simple call to fit
        return reg_pipe.fit(X_train, y_train)

    X_train_pre = reg_pipe[:-1].fit_transform(X_train, y_train)
    X_valid_pre = reg_pipe[:-1].transform(X_valid)

    # last step has to accept validation as a param
    reg_pipe[-1].set_params(validation=(X_valid_pre, y_valid))
    reg_pipe[-1].fit(X_train_pre, y_train)

    return reg_pipe


def cv_with_validation(estimator, X, y, cv, callbacks=None):
    """Perform cross-validation while passing validation data to the estimator
    in each fold.

    estimator is fit using `fit_with_validation`, and is assumed to be a pipeline that
    can accept a validation data parameter

    return results dictionary with one key per callback
    """
    cv = check_cv(cv, y, classifier=is_classifier(estimator))
    callbacks = callbacks if callbacks is not None else {}

    result = {k: {} for k in callbacks.keys()}
    result["train_time"] = {}

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        fold_estimator = clone(estimator)
        # form a dummy pipeline to make this compatible with single estimators
        if not isinstance(estimator, Pipeline):
            fold_estimator = make_pipeline("passthrough", fold_estimator)

        X_train, y_train = _safe_split(fold_estimator, X, y, train_idx)
        X_valid, y_valid = _safe_split(
            fold_estimator, X, y, test_idx, train_indices=train_idx
        )

        start_time = time()

        if "validation" in fold_estimator[-1].get_params():
            # pass validation data if validation param exists in final pipeline step
            fit_with_validation(fold_estimator, X_train, y_train, X_valid, y_valid)
        else:
            fit_with_validation(fold_estimator, X_train, y_train)

        fit_time = time() - start_time
        result["train_time"][fold] = fit_time

        if not isinstance(estimator, Pipeline):
            # revert back to single estimator for evaluation
            fold_estimator = fold_estimator[-1]

        for k, func in callbacks.items():
            result[k][fold] = func(
                fold=fold,
                estimator=fold_estimator,
                indices=(train_idx, test_idx),
                train_data=(X_train, y_train),
                test_data=(X_valid, y_valid),
            )
    return result


def _score_regressor(reg, X, y, eval_name=None):
    """score fitted classifier for common metrics.

    Applies metrics that compare true targets to predicted targets.
    """
    other_evals = {
        "rmse": partial(skm.mean_squared_error, squared=False),
        "r2": skm.r2_score,
        "mae": skm.mean_absolute_error,
        "mape": skm.mean_absolute_percentage_error,
    }

    y_pred = reg.predict(X)
    eval_vals = {s: f(y_pred, y) for s, f in other_evals.items()}
    if eval_name is not None:
        eval_vals["eval_set"] = eval_name

    return pd.DataFrame(pd.Series(eval_vals)).T


def _score_train(*, estimator, train_data, **kwargs):
    X, y = train_data
    return _score_regressor(estimator, X, y, eval_name="train")


def _score_test(*, estimator, test_data, **kwargs):
    X, y = test_data
    return _score_regressor(estimator, X, y, eval_name="test")


def _predict_test(*, estimator, test_data, **kwargs):
    X, y = test_data
    return estimator.predict(X)


def lgbm_fit_metrics(*, estimator, **kwargs):
    """Return fit metrics for fitted LightGBM model"""
    reg = estimator["regressor"]
    best_ntree = reg.best_iteration_ if reg.best_iteration_ else reg.n_estimators
    best_idx = best_ntree - 1

    lgbm_evals = {
        **{"train_" + k: v[best_idx] for k, v in reg.evals_result_["training"].items()},
        **{
            "test_" + k: v[best_idx] for k, v in reg.evals_result_["validation"].items()
        },
        **{"best_ntree": best_ntree},
    }
    return pd.DataFrame(pd.Series(lgbm_evals)).T


def common_cv_callbacks():
    """Generates dictionary of common CV callback functions for cv_with_validation.

    includes:
        - fitted estimator
        - fold indices
        - evaluation metrics on training data
        - evaluation metrics on test data
        - test predictions
        - test target data
    """
    callbacks = {
        "estimator": lambda *, estimator, **kw: estimator,
        "indices": lambda *, indices, **kw: indices,
        "eval_train": _score_train,
        "eval_test": _score_test,
        "predict_test": _predict_test,
        "y_test": lambda *, test_data, **kw: test_data[1],
    }
    return callbacks
