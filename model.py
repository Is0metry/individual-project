'''model contains helper functions to
assist in Modeling portion of final_report.ipynb'''
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from IPython.display import Markdown as md
from sklearn.cluster import KMeans
from sklearn.linear_model import LassoLars, LinearRegression, TweedieRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

import evaluate as ev
from custom_dtypes import LinearRegressionType, ModelDataType
from wrangle import scale


def select_baseline(ytrain: pd.Series) -> Tuple[md, pd.DataFrame]:
    '''tests mean and median of training data as a baseline metric.
    # Parameters
    ytrain: `pandas.Series` containing the target variable
    # Returns
    Formatted `Markdown` with information on best-performing baseline.
    '''
    med_base = ytrain.median()
    mean_base = ytrain.mean()
    mean_eval = ev.regression_errors(ytrain, mean_base, 'Mean Baseline')
    med_eval = ev.regression_errors(ytrain, med_base, 'Median Baseline')
    ret_md = pd.concat([mean_eval, med_eval]).to_markdown()
    ret_md += '\n### Because mean outperformed median on all metrics, \
        we will use mean as our baseline'
    return md(ret_md), mean_eval