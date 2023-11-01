"""
Simple routine to test GPU functionality
"""

import timeit

import cloudpickle
import cuml
import fire
import lightgbm as lgbm
import pandas as pd
import sklearn
import umap
from loguru import logger
from sklearn.datasets import make_regression

from housing import load_prep, model, utils


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
        "method": "fft" if use_cuml else "barnes_hut",
        "init": "random",
        "learning_rate": 200,
        "verbose": 6
    }
    if use_cuml:
        tsne_model = cuml.TSNE(**tsne_params)
    else:
        tsne_model = sklearn.manifold.TSNE(**tsne_params)

    result = tsne_model.fit_transform(X)
    logger.info((type(result), result.shape))
    return pd.DataFrame(
        result, index=X.index, columns=["TSNE1", "TSNE2"]
    )


def train_umap(X, use_cuml=True):
    umap_params = {
        "n_epochs": 10_000,
        "init": "random",
        "verbose": True
    }
    if use_cuml:
        umap_model = cuml.UMAP(**umap_params)
    else:
        umap_model = umap.UMAP(**umap_params)

    result = umap_model.fit_transform(X)
    logger.info((type(result), result.shape))
    return pd.DataFrame(
        result, index=X.index, columns=["UMAP1", "UMAP2"]
    )


def compare_embedding(outdir=None):
    raw_train_df, target_ds = load_prep.raw_train()
    X, y = raw_train_df, load_prep.transform_target(target_ds)
    pca_pipe = model.get_pca_pipeline(cat_n_components=10, num_n_components=10)
    pca_df = pca_pipe.fit_transform(X, y)

    fit_time, embedding = {}, {}
    logger.info('training tsne cpu')
    fit_time['tsne_cpu'] = timeit.repeat(lambda: train_tsne(pca_df, use_cuml=False), number=1)
    embedding['tsne_cpu'] = train_tsne(pca_df, use_cuml=False)
    
    logger.info('training tsne gpu')
    fit_time['tsne_gpu'] = timeit.repeat(lambda: train_tsne(pca_df, use_cuml=True), number=1)
    embedding['tsne_gpu'] = train_tsne(pca_df, use_cuml=True)

    logger.info('training umap cpu')
    fit_time['umap_cpu'] = timeit.repeat(lambda: train_umap(pca_df, use_cuml=False), number=1)
    embedding['umap_cpu'] = train_umap(pca_df, use_cuml=False)
    
    logger.info('training umap gpu')
    fit_time['umap_gpu'] = timeit.repeat(lambda: train_umap(pca_df, use_cuml=True), number=1)
    embedding['umap_gpu'] = train_umap(pca_df, use_cuml=True)

    print(fit_time)
    if outdir is not None:
        with open(utils.WORKING_DIR / outdir / "embedding_fit_times.pkl", "wb") as f:
            cloudpickle.dump(fit_time, f)
        with open(utils.WORKING_DIR / outdir / "embedding.pkl", "wb") as f:
            cloudpickle.dump(embedding, f)


if __name__ == "__main__":
    fire.Fire()
