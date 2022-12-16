'''contains the data type aliases used in typing of each function for docstrings'''
import typing as t

import pandas as pd
from numpy.typing import ArrayLike
from sklearn.linear_model import LassoLars, LinearRegression, TweedieRegressor
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from wood_steel_regressor import WoodSteelRegression

ModelDataType = t.Union[ArrayLike, pd.DataFrame, pd.Series]
LinearRegressionType = t.Union[LinearRegression, LassoLars,
                               TweedieRegressor, WoodSteelRegression]
PandasDataType = t.Union[pd.Series, pd.DataFrame]
ScalerType = t.Union[MinMaxScaler, RobustScaler]
lmplot_kwargs = {'scatter': {'color': '#40b7ad'}, 'line': {'color': '#2e1e3b'}}
