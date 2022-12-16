'''model contains helper functions to
assist in Modeling portion of final_report.ipynb'''
from copy import deepcopy
from itertools import product
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from IPython.display import Markdown as md
from sklearn.cluster import KMeans
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LassoLars, LinearRegression, TweedieRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures, RobustScaler

import evaluate as ev
from custom_dtypes import LinearRegressionType, ModelDataType, ScalerType
from wood_steel_regressor import WoodSteelRegression


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
    ret_md += ('\n### Because mean outperformed median on all metrics,'
               'we will use mean as our baseline')
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
        ret_scaled = pd.DataFrame(
            scaler.transform(ret_scaled), columns=df.columns)
    except NotFittedError:
        scaler = scaler.fit(ret_scaled)
        ret_scaled = pd.DataFrame(
            scaler.transform(ret_scaled), columns=df.columns)
    return ret_scaled


def cluster(df: pd.DataFrame,
            kmeans: KMeans) -> pd.Series:
    # TODO docstring
    clusters = np.array([])
    try:
        clusters = kmeans.predict(df)
    except NotFittedError:
        clusters = kmeans.fit_predict(df)
    return pd.Series(clusters)


def train_and_validate(train: pd.DataFrame,
                       validate: pd.DataFrame,
                       features: List[str],
                       target: str,
                       regressor: LinearRegressionType,
                       name: str,
                       poly_scaler: Union[PolynomialFeatures,
                                          None] = None) -> pd.DataFrame:
    # TODO docstring
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
    if isinstance(regressor, WoodSteelRegression):
        strain = pd.concat([strain, train.steel_track], axis=1)
        svalid = pd.concat([svalid, validate.steel_track], axis=1)
    trainy_pred = run_regression(strain, regressor, trainy[target])
    validy_pred = run_regression(svalid, regressor)
    train_mse = ev.root_mean_squared_error(trainy[target], trainy_pred)
    validate_mse = ev.root_mean_squared_error(validy[target], validy_pred)
    ret_df = pd.DataFrame(
        {'Train': train_mse, 'Validate': validate_mse}, index=[name])
    return ret_df


def get_baseline(y_true: pd.Series, y_pred: float) -> float:
    df_base = np.array([y_pred for i in range(y_true.shape[0])])
    return ev.root_mean_squared_error(y_true, df_base)


def try_models_train_validate(train: pd.DataFrame,
                              validate: pd.DataFrame) -> pd.DataFrame:
    # TODO Docstring
    def tv_run(r, n, p=None): return train_and_validate(
        train, validate, ['speed', 'man_group', 'height', 'num_inversions'],
        'length', r, n, p)

    linreg = tv_run(LinearRegression(), 'Linear Regression')
    linreg_square = tv_run(
        LinearRegression(), 'Squared Regression', PolynomialFeatures(2))
    linreg_cube = tv_run(LinearRegression(),
                         'Cubed Regression', PolynomialFeatures(3))
    llars = tv_run(LassoLars(alpha=2.0), 'LASSO+LARS')
    llars_square = tv_run(LassoLars(alpha=2.0),
                          'LASSO+LARS^2', PolynomialFeatures(2))
    llars_cube = tv_run(LassoLars(alpha=2.0),
                        'LASSO+LARS^3', PolynomialFeatures(3))
    glm = tv_run(TweedieRegressor(power=1, alpha=1), 'GLM')
    glm_square = tv_run(TweedieRegressor(power=1, alpha=1),
                        'GLM^2', PolynomialFeatures(2))
    glm_cube = tv_run(TweedieRegressor(power=1, alpha=1),
                      'GLM^3', PolynomialFeatures(3))
    baseline = train.length.mean()
    baseline = pd.DataFrame({'Train': get_baseline(train.length,
                                                   baseline),
                             'Validate': get_baseline(
        validate.length, baseline)}, index=['Baseline'])
    return pd.concat([linreg, linreg_square, linreg_cube, llars, llars_square,
                      llars_cube, glm, glm_square, glm_cube, baseline])


def wood_steel_permutations(
        train: pd.DataFrame,
        validate: pd.DataFrame) -> pd.DataFrame:
    # TODO Docstring
    def tv_run(r, n, p=None): return train_and_validate(
        train, validate, ['speed', 'man_group',
                          'height', 'num_inversions'],
        'length', r, n, p)
    regressors = [(LinearRegression(), 'LinearRegression'), (LassoLars(
        alpha=4), 'LassoLars'), (TweedieRegressor(power=0, alpha=4), 'GLM')]
    mse = []
    for reg in product(regressors, repeat=2):
        reg1 = reg[0]
        reg2 = reg[1]
        linear = WoodSteelRegression(deepcopy(reg1[0]), deepcopy(reg2[0]))
        squared = WoodSteelRegression(deepcopy(reg1[0]), deepcopy(reg2[0]))
        cubed = WoodSteelRegression(deepcopy(reg1[0]), deepcopy(reg2[0]))
        mse.append(tv_run(linear, reg1[1] + '+' + reg2[1]))
        mse.append(
            tv_run(squared,
                   reg1[1] + '+' + reg2[1] + '^2',
                   PolynomialFeatures(2)))
        mse.append(tv_run(cubed, reg1[1] + '+' +
                   reg2[1] + '^3', PolynomialFeatures(3)))
    ret_frame = pd.concat(mse).sort_values(
        by=['Validate', 'Train']).iloc[:10, :]
    return ret_frame


def run_test(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    # TODO docstring
    features = ['speed', 'man_group', 'height', 'num_inversions']
    testx = test[features]
    testy = test.length
    trainx = train[features]
    trainy = train.length
    scaler = RobustScaler()
    trainx = scaler.fit_transform(trainx, trainy)
    stest = scale(testx, scaler)
    poly = PolynomialFeatures(2)
    trainx = poly.fit_transform(trainx)
    stest = poly.transform(stest)
    linreg = LinearRegression()
    linreg.fit(trainx, trainy)
    ypred = run_regression(stest, linreg)
    test_rmse = ev.root_mean_squared_error(test.length, ypred)
    baseline = get_baseline(test.length, train.length.mean())
    ret = pd.DataFrame({'LinearRegression': test_rmse,
                       'Baseline': baseline}, index=['Test']).T
    return ret
