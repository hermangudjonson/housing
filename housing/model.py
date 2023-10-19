"""
Routines for housing model generation and evaluation.
"""

import numpy as np
import pandas as pd
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.decomposition import PCA
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge, ElasticNetCV
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.naive_bayes import CategoricalNB
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline, make_union
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
)

from housing import load_prep


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


def get_regressor(strategy="passthrough"):
    if strategy == "passthrough":
        return "passthrough"
    elif strategy == "linear":
        return ElasticNetCV(l1_ratio=0.9)
    else:
        raise ValueError(f"unimplemented strategy {strategy}")


def get_reg_pipeline(reg_strategy="passthrough", onehot_encode=False, include_pca=True):
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

    reg_pipe = Pipeline(
        [
            ("preprocess", load_prep.preprocess_pipe()),
            ("impute", impute_step),
            ("PCA", pca_step if include_pca else "passthrough"),
            ("onehot", impute_to_onehot if onehot_encode else "passthrough"),
            ("regressor", get_regressor(reg_strategy)),
        ]
    )
    # since we are passing info between steps, we override clone
    reg_pipe.__sklearn_clone__ = lambda: get_reg_pipeline(
        reg_strategy, onehot_encode, include_pca
    )
    return reg_pipe
