"""
Simple routine to test GPU functionality
"""

import lightgbm as lgbm
import sklearn
from sklearn.datasets import make_regression
import cuml
import timeit
import fire
from loguru import logger


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


def compare_device(gpu_device='gpu'):
    X, y = make_regression(n_samples=1_000_000, n_features=100, n_informative=10)

    logger.info('training linear cpu')
    print(timeit.repeat(lambda x: train_linear(X, y, use_cuml=False), number=1))
    logger.info('training linear gpu')
    print(timeit.repeat(lambda x: train_linear(X, y, use_cuml=True), number=1))
    logger.info('training lightgbm cpu')
    print(timeit.repeat(lambda x: train_lightgbm(X, y, device='cpu'), number=1))
    logger.info('training lightgbm gpu')
    print(timeit.repeat(lambda x: train_lightgbm(X, y, device=gpu_device), number=1))


if __name__ == "__main__":
    fire.Fire()
