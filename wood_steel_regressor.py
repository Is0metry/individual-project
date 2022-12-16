import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import (
    LassoLars, LinearRegression, TweedieRegressor)
from typing import Union,TypeVar
from numpy.typing import NDArray, ArrayLike

T = TypeVar('T')

class WoodSteelRegression:
    ''' a regressor that runs a separate regression model on both wood and steel '''
    RegressionType = Union[LinearRegression, TweedieRegressor, LassoLars]
    steel_regressor: RegressionType
    wood_regressor: RegressionType

    def __init__(self:T, s_regressor: RegressionType,
                 w_regression: RegressionType) -> None:
        self.steel_regressor = s_regressor
        self.wood_regressor = w_regression

    def fit(self:T, x: pd.DataFrame, y: pd.Series) -> T:
        # TODO docstring
        y.index = x.index
        x_steel = x[x.steel_track].drop(columns=['steel_track'])
        x_wood = x[~x.steel_track].drop(columns=['steel_track'])
        y_steel = y.iloc[x_steel.index]
        y_wood = y.iloc[y_steel.index]
        self.wood_regressor = self.wood_regressor.fit(x_wood, y_wood)
        self.steel_regressor = self.steel_regressor.fit(x_steel, y_steel)
        return self

    def predict(self:T, x: pd.DataFrame) -> ArrayLike:
        # TODO Docstring
        x_steel = x[x.steel_track].drop(columns=['steel_track'])
        x_wood = x[~x.steel_track].drop(columns=['steel_track'])
        ret_steel = pd.Series(self.steel_regressor.predict(
            x_steel), index=x_steel.index)
        ret_wood = pd.Series(self.wood_regressor.predict(
            x_wood), index=x_wood.index)
        ret_series = pd.concat([ret_wood, ret_steel], axis=0).sort_index()
        return ret_series.to_numpy()
