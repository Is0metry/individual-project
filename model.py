'''model contains helper functions to
assist in Modeling portion of final_report.ipynb'''
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from IPython.display import Markdown as md
from sklearn.cluster import KMeans
from sklearn.linear_model import LassoLars, LinearRegression, TweedieRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from sklearn.exceptions import NotFittedError
import evaluate as ev
from custom_dtypes import LinearRegressionType, ModelDataType, ScalerType


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


def run_regression(df: pd.DataFrame,
                   regressor: LinearRegressionType,
                   target: Union[pd.Series, None] = None)\
        -> pd.Series:
    try:
        rets = pd.Series(regressor.predict(df))
        return rets
    except NotFittedError:
        if target is None:
            raise Exception('Target cannot be None if regressor is not fitted')
        regressor.fit(df, target)
        rets = pd.Series(regressor.predict(df))
        return rets


def scale(df: pd.DataFrame,
          scaler: ScalerType,
          ) -> pd.DataFrame:
    # TODO docstring
    ret_scaled = df.copy()
    try:
        ret_scaled = scaler.transform(ret_scaled)
    except NotFittedError:
        scaler = scaler.fit(ret_scaled)
        ret_scaled = pd.DataFrame(
            scaler.transform(ret_scaled), columns=df.columns)
    return ret_scaled


def cluster(df: pd.DataFrame,
            kmeans: KMeans) -> Union[pd.Series, KMeans]:
    # TODO docstring
    clusters = np.array([])
    try:
        clusters = kmeans.predict(df)
    except NotFittedError:
        clusters = kmeans.fit_predict(df)
    return pd.Series(clusters), kmeans


def train_and_validate(train: pd.DataFrame,
                       validate: pd.DataFrame,
                       features: List[str],
                       target: str,
                       regressor: LinearRegressionType,
                       name: str,
                       poly_scaler: Union[PolynomialFeatures,
                                          None] = None) -> pd.DataFrame:
    scaler = RobustScaler()
    trainx = train[features]
    trainy = train[[target]]
    validx = validate[features]
    validy = validate[[target]]
    scaler = RobustScaler()
    strain = scale(trainx, scaler)
    svalid = scale(validx, scaler)
    regressor = LinearRegression()
    if poly_scaler is not None:
        strain = pd.DataFrame(poly_scaler.fit_transform(strain, trainy))
        svalid = pd.DataFrame(poly_scaler.transform(svalid))
    trainy_pred = run_regression(strain, regressor, trainy[target])
    validy_pred = run_regression(svalid, regressor)
    train_rmse = ev.root_mean_squared_error(trainy[target], trainy_pred)
    validate_rmse = ev.root_mean_squared_error(validy[target], validy_pred)
    ret_df = pd.DataFrame(
        {'Train': train_rmse, 'Validate': validate_rmse}, index=[name])
    return ret_df
