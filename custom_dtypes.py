'''contains the data type aliases used in 
typing of each function for docstrings'''
import typing as t

import pandas as pd
from numpy.typing import ArrayLike
from sklearn.linear_model import LassoLars, LinearRegression, TweedieRegressor
from sklearn.preprocessing import MinMaxScaler, RobustScaler

ModelDataType = t.Union[ArrayLike, pd.DataFrame, pd.Series]
LinearRegressionType = t.Union[LinearRegression, LassoLars, TweedieRegressor]
ScalerType = t.Union[MinMaxScaler, RobustScaler]
