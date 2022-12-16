import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans

import explore as e
import model as m
import wrangle as w


def manufacturers_v_length(train: pd.DataFrame) -> None:
    sns.boxplot(data=train, x='manufacturer', y='length')
    plt.xticks(rotation=90)
    plt.show()


sns.barplot(data=train, x='steel_track', y='length')


# %% [markdown]
# ### Is there a positive relationship between material type and length?
# #### $H_0$: $\mu_{steel} \geq \mu_{wood} \geq \mu_{hybrid}$
# #### $H_a$: $\mu_{steel} < \mu_{wood} < \mu_{hybrid}$
#
# ### Assumptions:
# 1. sets are independent
# 2. sets are normally distributed

# %%
sns.histplot(data=train, x='length', hue='steel_track')


# %% [markdown]
#

# %%
steel = train[train.steel_track == 'Steel']
wood = train[train.steel_track == 'Wooden']
steel.length.var(), wood.length.var()


# %%
stats.levene(steel.length, wood.length)


# %%
t, p = stats.ttest_ind(steel.length, wood.length, equal_var=False)
e.t_to_md(p, t)


# %%
sns.lmplot(data=train, x='height', y='length', line_kws={'color': 'red'})


# %%
train.columns


# %%
e.generate_elbow(train[['height', 'speed']])


# %%
sns.lmplot(data=train, x='speed', y='length')


# %%
sns.scatterplot(data=train, x='speed', y='length', hue='steel_track')


# %% [markdown]
# # Observations
# - Length
# - Height
# - Speed
# - Manufacturer
# - Track Construction
#

# %%
train.manufacturer = train.manufacturer.astype('string')
grouping = train.groupby('manufacturer').mean().sort_values(
    by='length', ascending=False).index.to_list()
train['man_group'] = -1
for i in range(5):
    train.loc[train.manufacturer.isin(grouping[i*5:(i+1)*5]), 'man_group'] = i
sns.boxplot(data=train, x='man_group', y='length')


# %% [markdown]
# #### Question: is there a statistical difference in the means of the man_group?
# $H_0$: $\mu_{0} = \mu_1 = \mu_2 = \mu_3 = \mu_4$
#
# $H_a$: $\mu_{0} = \mu_1 \neq \mu_2 \neq \mu_3 \neq \mu_4$

# %%
man_groups = []
for i in range(5):
    ser = train[train.man_group == i].length
    man_groups.append(ser)
t, p = stats.f_oneway(*man_groups)
e.t_to_md(p, t)


# %%
sns.relplot(data=train, x='speed', y='length',
            hue='man_group', palette='colorblind')

# %%
reload(w)
train['height_speed'], kmeans = w.cluster(
    train[['speed', 'man_group']], kmeans=KMeans(7))


# %%
sns.boxplot(data=train, x='height_speed', y='length')

# %%
reload(e)
t, p = stats.levene(train.speed, train.length)
e.t_to_md(p, t)

# %%
r, p = stats.spearmanr(train.speed, train.length)
e.p_to_md(p, r=r)

# %%
md, base = m.select_baseline(train.length)
md

# %%
