"""
LightGBM hyperparameter optimization with optuna

Define optuna objective, study and runs.
Additionally compare GPU fitting performance.

tuning progression:

"""

from housing import load_prep, model, utils

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
import optuna
import cloudpickle

import functools
import warnings
import fire





if __name__ == "__main__":
    fire.Fire()
