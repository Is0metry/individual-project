import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import Markdown as md
from scipy import stats
from sklearn.cluster import KMeans
from model import cluster


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


def executive_summary(train: pd.DataFrame) -> None:
    '''
    Produces a pair of graphs for the executive summary
    ## Parameters
    train: `DataFrame` with training data
    ## Returns
    None, plots the plots for the executive summary to notebook
    '''
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    sns.barplot(data=train, x='manufacturer', y='length',
                palette='mako', errorbar=None, ax=axs[0])

    axs[0].set_xticklabels(axs[0].get_xticklabels(
    ), rotation=35, horizontalalignment='right', fontsize='xx-small')
    axs[0].set_xlabel('Manufacturer')
    axs[0].set_ylabel('Average Length')
    axs[0].set_title('Coaster Length by Manufacturer')
    sns.scatterplot(data=train, x='speed', y='length',
                    palette='mako', ax=axs[1])
    axs[1].set_title('Speed vs. Length')
    axs[1].set_xlabel('Speed')
    axs[1].set_ylabel('Length')
    plt.show()


def manufacturers_v_length(train: pd.DataFrame) -> None:
    '''
    Plots the average coaster length by manufacturer
    ## Parameters
    train: `DataFrame` containing the training data
    ## Returns
    None, plots to the notebook
    '''
    sns.barplot(data=train, x='manufacturer', y='length',
                palette='mako', errorbar=None)
    plt.xticks(rotation=35, horizontalalignment='right', fontsize='x-small')
    plt.xlabel('Manufacturer')
    plt.ylabel('Average Length')
    plt.title('Coaster Length by Manufacturer')
    plt.show()


def man_speed_kde(train: pd.DataFrame) -> None:
    '''
    Plots a KDE graph comparing the clusters based on
    manufacturer group and the speed
    ## Parameters
    train: `DataFrame` containing clustered training
    data
    ## Returns
    None: plots to the notebook
    '''
    train['man_speed_cluster'] = cluster(
        train[['speed', 'num_inversions']], KMeans(6))
    sns.kdeplot(data=train, x='length',
                hue='man_speed_cluster', palette='mako')
    plt.show()


def wood_vs_steel(train: pd.DataFrame) -> None:
    # TODO Docstring
    sns.boxplot(data=train, x='steel_track', y='length', palette='mako')
    plt.xticks(rotation=35)
    plt.title('Wood vs. Steel Track Length')
    plt.xlabel('Track Material')
    plt.ylabel('Coaster Length')
    plt.show()


def wood_steel_levene(train: pd.DataFrame) -> md:
    '''
    Performs a levene test to check variance
    on steel and wooden track
    ## Parameters
    train: `DataFrame` containing the training
    data
    ## Returns
    a `Markdown` object containing t and p
    values for the levene test, as well as a
    statement on whether or not the
    null hypothesis is rejected
    '''
    steel = train[train.steel_track].length
    wood = train[~train.steel_track].length
    t, p = stats.levene(steel, wood)
    return t_to_md(p, t)


def wood_steel_ttest(train: pd.DataFrame) -> md:
    '''
    Performs an independent test to check variance
    on steel and wooden track
    ## Parameters
    train: `DataFrame` containing the training
    data
    ## Returns
    a `Markdown` object containing t and p
    values for the statistical test, as well as a
    statement on whether or not the
    null hypothesis is rejected
    '''
    steel = train[train.steel_track].length
    wood = train[~train.steel_track].length
    t, p = stats.ttest_ind(wood, steel, equal_var=False)
    return t_to_md(p, t)


def speed_length(train: pd.DataFrame) -> None:
    '''
    Plots a scatterplot of coaster speed
    vs coaster length
    ## Parameters
    train: `DataFrame containing training data
    ## Returns
    None, plots scatterplot to the notebook
    '''
    sns.scatterplot(data=train, x='speed', y='length', palette='mako')
    plt.title('Speed vs. Length')
    plt.xlabel('Speed')
    plt.ylabel('Length')
    plt.show()


def speed_len_boxplot(train: pd.DataFrame) -> None:
    sns.boxplot(data=train[['speed', 'length']], palette='mako')


def speed_len_spearmanr(train: pd.DataFrame) -> md:
    r, p = stats.spearmanr(train.speed, train.length)
    return p_to_md(p, r=r)
