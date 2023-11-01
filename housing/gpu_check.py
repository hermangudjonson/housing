"""
Simple routine to test GPU functionality
"""

import timeit

import cuml
import fire
import lightgbm as lgbm
import pandas as pd
import sklearn
from loguru import logger
from sklearn.datasets import make_regression
import umap


def train_lightgbm(X, y, device="gpu"):
    lgbm_params = {
        "objective": "regression",
        "max_bin": 63,
        "num_leaves": 255,
        "n_estimators": 50,
        "learning_rate": 0.1,
        "device": device,
        "gpu_platform_id": 0,
        "gpu_device_id": 0,
        "verbose": 1,
    }

    lgbm_model = lgbm.LGBMRegressor(**lgbm_params)
    lgbm_model.fit(X, y)
    return lgbm_model


def train_linear(X, y, use_cuml=True):
    if use_cuml:
        lin_model = cuml.linear_model.LinearRegression()
    else:
        lin_model = sklearn.linear_model.LinearRegression()
    lin_model.fit(X, y)
    return lin_model


def compare_device(gpu_device="gpu"):
    X, y = make_regression(n_samples=1_000_000, n_features=100, n_informative=10)

    logger.info("training linear cpu")
    print(timeit.repeat(lambda: train_linear(X, y, use_cuml=False), number=1))
    logger.info("training linear gpu")
    print(timeit.repeat(lambda: train_linear(X, y, use_cuml=True), number=1))
    logger.info("training lightgbm cpu")
    print(timeit.repeat(lambda: train_lightgbm(X, y, device="cpu"), number=1))
    logger.info("training lightgbm gpu")
    print(timeit.repeat(lambda: train_lightgbm(X, y, device=gpu_device), number=1))


def train_tsne(X, use_cuml=True):
    # params should be valid for both sklearn and cuml
    tsne_params = {
        "n_iter": 10_000,
        "method": "barnes_hut",
        "init": "random",
        "learning_rate": 200,
    }
    if use_cuml:
        tsne_model = cuml.TSNE(**tsne_params)
    else:
        tsne_model = sklearn.manifold.TSNE(**tsne_params)

    return pd.DataFrame(
        tsne_model.fit_transform(X), index=X.index, columns=["TSNE1", "TSNE2"]
    )


def train_umap(X, use_cuml=True):
    umap_params = {
        "n_epochs": 10_000,
        "init": "random",
    }
    if use_cuml:
        umap_model = cuml.UMAP(**umap_params)
    else:
        umap_model = umap.UMAP(**umap_params)

    return pd.DataFrame(
        umap_model.fit_transform(X), index=X.index, columns=["UMAP1", "UMAP2"]
    )


def compare_embedding():
    pass


if __name__ == "__main__":
    fire.Fire()
