"""Routines for loading and preprocessing housing data
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, PowerTransformer, StandardScaler

from housing.utils import INPUT_DIR


def raw_train(input_dir: Path | str = INPUT_DIR):
    """Load housing raw training data

    Parameters
    ----------
    input_dir : Path | str, optional
        directory containing csv data, by default INPUT_DIR

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        tuple containing:
         - 1460 x 79 training dataframe with index Id
         - 1460 target data series with index Id and name SalePrice
    """
    input_dir = Path(input_dir) if isinstance(input_dir, str) else input_dir

    raw_train_df = pd.read_csv(input_dir / "train.csv", index_col="Id")
    train_df = raw_train_df.drop(columns="SalePrice")
    target_ds = raw_train_df["SalePrice"]

    return train_df, target_ds


def raw_test(input_dir: Path | str = INPUT_DIR):
    """Load housing raw test data

    Parameters
    ----------
    input_dir : Path | str, optional
        directory containing csv data, by default INPUT_DIR

    Returns
    -------
    pd.DataFrame
        1459 x 79 test dataframe with index Id
    """
    input_dir = Path(input_dir) if isinstance(input_dir, str) else input_dir

    test_df = pd.read_csv(input_dir / "test.csv", index_col="Id")
    return test_df


def example_submission(input_dir: Path | str = INPUT_DIR):
    """Load housing example submission

    Parameters
    ----------
    input_dir : Path | str, optional
        directory containing csv data, by default INPUT_DIR

    Returns
    -------
    pd.Series
        1459 series containing example predictions on test data
    """
    input_dir = Path(input_dir) if isinstance(input_dir, str) else input_dir

    submission_ds = pd.read_csv(
        input_dir / "sample_submission.csv", index_col="Id"
    ).squeeze()
    return submission_ds


def transform_target(y):
    return np.log(y)


def inv_transform_target(y):
    return np.exp(y)


def near_zero_var(
    X: pd.DataFrame, *, freq_ratio=20, unique_pct=0.1, plot=False, return_stats=False
) -> pd.Series | pd.DataFrame:
    """Evaluates each feature for near zero variance.

    Feature is near zero variance if ratio of largest freq value
    to 2nd largest freq value is high and percentage of unique values is low

    Parameters
    ----------
    X : pd.DataFrame
    freq_ratio : int, optional
        threshold for ratio of largest freq value to 2nd largest freq value, by default 20
    unique_pct : float, optional
        threshold for percentage of unique values to total samples, by default 0.1
    plot : bool, optional
        plot feature stats, by default False
    return_stats : bool, optional
        return dataframe with stats if True, otherwise just return boolean series, by default False

    Returns
    -------
    pd.Series | pd.DataFrame
        boolean series indicating near zero var, otherwise dataframe with stats
    """

    def _freq_ratio(col):
        col_counts = col.value_counts(dropna=False)
        return (
            col_counts.iloc[0] / col_counts.iloc[1] if len(col_counts) > 1 else math.inf
        )

    def _unique_pct(col):
        return col.nunique() / len(col)

    stat_agg = X.agg([_freq_ratio, _unique_pct], axis=0).T
    nzvar = (stat_agg._freq_ratio > freq_ratio) & (stat_agg._unique_pct < unique_pct)
    stat_agg["near_zero_var"] = nzvar
    logger.info("{total} columns with near zero var", total=sum(nzvar))
    if return_stats:
        nzvar = stat_agg

    if plot:
        sns.scatterplot(stat_agg, x="_unique_pct", y="_freq_ratio", hue="near_zero_var")
        plt.xscale("log")
        plt.yscale("log")
        plt.show()

    return nzvar


class NearZeroVarSelector(BaseEstimator, TransformerMixin):
    def __init__(self, freq_ratio=20, unique_pct=0.1):
        self.freq_ratio = freq_ratio
        self.unique_pct = unique_pct

    def fit(self, X, y=None):
        self.nzvar_ = near_zero_var(
            X, freq_ratio=self.freq_ratio, unique_pct=self.unique_pct
        )
        return self

    def transform(self, X):
        return X.loc[:, ~self.nzvar_]


def preprocess_pipe():
    """Pipeline that filters near-zero variance columns and scales numeric data

    Returns
    -------
    Pipeline
    """
    cat_scale = make_column_transformer(
        (
            # convert any object columns to categoricals
            FunctionTransformer(
                lambda X: X.astype("category"), feature_names_out="one-to-one"
            ),
            make_column_selector(dtype_include="object"),
        ),
        # power transform all numeric columns
        remainder=make_pipeline(StandardScaler(), PowerTransformer()).set_output(
            transform="pandas"
        ),
        verbose_feature_names_out=False
    ).set_output(transform='pandas')

    return make_pipeline(NearZeroVarSelector(), cat_scale)