import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import Markdown as md
from scipy import stats
from sklearn.cluster import KMeans
from wrangle import cluster

def p_to_md(p: float, alpha: float = .05, **kwargs) -> md:
    '''
    returns the result of a p test as a `Markdown` object
    ## Parameters
    p: `float` of the p value from performed Hypothesis test
    alpha: `float` of alpha value for test, defaults to 0.05
    kwargs: any additional return values of statistical test
    ## Returns
    formatted `Markdown` object containing results of hypothesis test.

    '''
    ret_str = ''
    p_flag = p < alpha
    ret_str += f'## p = {p}\n'
    for k, v in kwargs.items():
        ret_str += f'## {k} = {v}\n\n'
    ret_str += (f'## Because $\\alpha$ {">" if p_flag else "<"} p,'
        f'we {"failed to " if ~(p_flag) else ""} reject $H_0$')
    return md(ret_str)


def t_to_md_1samp(p: float, t: float, alpha: float = .05, **kwargs):
    '''takes a p-value, alpha, and any T-test arguments and
    creates a Markdown object with the information.
    ## Parameters
    p: float of the p value from run T-Test
    t: float of the t-value from run T-Test
    alpha: desired alpha value, defaults to 0.05
    ## Returns
    `IPython.display.Markdown` object with results of the statistical test
    '''
    ret_str = ''
    t_flag = t > 0
    p_flag = p/2 < alpha
    ret_str += f'## t = {t} \n\n'
    for k, v in kwargs.items():
        ret_str += f'## {k} = {v}\n\n'
    ret_str += f' ## p/2 = {p/2} \n\n'
    ret_str += (f'## Because t {">" if t_flag else "<"} 0 '
                f'and $\\alpha$ {">" if p_flag else "<"} p/2, '
                f'we {"failed to " if ~(t_flag & p_flag) else ""} '
                ' reject $H_0$')
    return md(ret_str)


def t_to_md(p: float, t: float, alpha: float = .05, **kwargs):
    '''takes a p-value, alpha, and any T-test arguments and
    creates a Markdown object with the information.
    ## Parameters
    p: float of the p value from run T-Test
    t: float of the t-value from run T-Test
    alpha: desired alpha value, defaults to 0.05
    ## Returns
    `IPython.display.Markdown` object with results of the statistical test
    '''
    ret_str = ''
    t_flag = t > 0
    p_flag = p < alpha
    ret_str += f'## t = {t} \n\n'
    for k, v in kwargs.items():
        ret_str += f'## {k} = {v}\n\n'
    ret_str += f' ## p = {p} \n\n'
    ret_str += (f'## Because t {">" if t_flag else "<"} 0 '
                f'and $\\alpha$ {">" if p_flag else "<"} p, '
                f'we {"failed to " if ~(t_flag & p_flag) else ""} '
                ' reject $H_0$')
    return md(ret_str)


def generate_elbow(df: pd.DataFrame, k_min: int = 1, k_max: int = 30) -> None:
    '''
    Plots KMeans elbow of a given potential cluster as well as the
    percent change for the graph
    ## Parameters
    df: `DataFrame` containing features to perform KMeans clustering on
    k_min: `int` specifying minimum number of centroids
    k_max: `int` specifying maximum number of centroids
    ## Returns
    None (plots graph to Jupyter notebook)
    '''
    with plt.style.context('seaborn-whitegrid'):
        inertia = {i: KMeans(i, random_state=420).fit(
            df).inertia_ for i in range(k_min, k_max)}
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        sns.lineplot(data=inertia, ax=axs[0])
        axs[0].set_title('Inertia')
        axs[0].set_xlabel('No. of Clusters')
        axs[0].set_ylabel('Inertia')
        pct_change = [((inertia[i]-inertia[i+1])/inertia[i])
                      * 100 for i in range(k_min, k_max-1)]
        sns.lineplot(data=pct_change, ax=axs[1])
        axs[1].set_xlabel('No. of Clusters')
        axs[1].set_ylabel('% of Change')
        axs[1].set_title('% Change')
        fig.tight_layout()
        plt.show()
