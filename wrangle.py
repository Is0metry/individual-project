import numpy as np
import pandas as pd
from typing import List, Tuple, Union
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.exceptions import NotFittedError
from custom_dtypes import ScalerType
from sklearn.cluster import KMeans


def acquire_coasters() -> pd.DataFrame:
    '''
    Acquires the coaster data from CSV
    ## Parameters
    None
    ## Returns
    `DataFrame` containing coaster information
    '''
    coaster_df = pd.read_csv('data/prepped/roller_coasters.csv')
    return coaster_df


def mark_other_manufacturers(manufacturer, others=[]):
    # TODO docstring
    if manufacturer in others:
        return "Other"
    elif manufacturer == 'Philadelphia Toboggan Coaster':
        return 'PTC'
    return manufacturer


def prepare_coasters(coaster_df: pd.DataFrame) -> pd.DataFrame:
    # TODO Docstring
    # Update some...oversights
    # info pulled from RCDB
    coaster_df.iloc[2591, :] = pd.Series({'name': 'Orion',
                                         'material_type': 'Steel',
                                          'seating_type': 'Sit Down',
                                          'speed': 146.45,
                                          'height': 87.48,
                                          'length': 1621.84,
                                          'num_inversions': 0,
                                          'manufacturer': 'B&M',
                                          'park': 'Kings Island',
                                          'status': 'operating'})
    coaster_df.iloc[2592, :] = pd.Series({'name': 'Ice Breaker',
                                          'material_type': 'Steel',
                                          'seating_type': 'Sit Down',
                                          'speed': 83.69, 'height': 28.35,
                                          'length': 579.12,
                                          'num_inversions': 0,
                                         'manufacturer': 'Premier Rides',
                                          'park': 'SeaWorld Orlando',
                                          'status': 'operating'})
    coaster_df.iloc[2669, :] = pd.Series({'name': 'Velocicoaster',
                                         'material_type': 'Steel',
                                          'seating_type': 'Sit Down',
                                          'speed': 112.65,
                                          'height': 47.24,
                                          'length': 1432.56,
                                          'num_inversions': 4,
                                          'manufacturer': 'Intamin',
                                          'park': 'Universal Studios'
                                          'Islands of Adventure',
                                          'status': 'operating'})
    coaster_df.iloc[2668, :] = pd.Series({'name': 'Texas Stingray',
                                          'material_type': 'Hybrid',
                                          'seating_type': 'Sit Down',
                                          'speed': 88.51,
                                         'height': 30.48,
                                          'length': 1029.92,
                                          'num_inversions': 0,
                                          'manufacturer': 'GCI',
                                          'park': 'SeaWorld San Antonio',
                                          'status': 'operating'})
    coaster_df.iloc[2601, :] = pd.Series({'name': 'Namazu',
                                          'material_type': 'Steel',
                                          'seating_type': 'Sit Down',
                                          'speed': 84.49,
                                         'height': 15.85,
                                          'length': 584.00,
                                          'num_inversions': 0,
                                          'manufacturer': 'Intamin',
                                          'park': 'Vulcania',
                                          'status': 'operating'})
    coaster_df.loc[coaster_df.name ==
                   'Escape from Madagascar', 'material_type'] = 'Steel'
    coaster_df.iloc[[683, 730], 4] = 9.2
    coaster_df = coaster_df[coaster_df.status != 'rumored']
    coaster_df = coaster_df[coaster_df.material_type != 'na']
    coaster_df = coaster_df[coaster_df.manufacturer != 'na']
    coaster_df = coaster_df.dropna()
    counts = coaster_df.manufacturer.value_counts()
    counts = counts[counts < 8].index.format()
    coaster_df.manufacturer = coaster_df.manufacturer.apply(
        mark_other_manufacturers, others=counts)
    coaster_df.seating_type = coaster_df.seating_type.astype('category')
    arrow_hybrid = (coaster_df.material_type == 'Hybrid') &\
        (coaster_df.manufacturer == 'Arrow')
    rmc_hybrid = (coaster_df.material_type == 'Hybrid') &\
        (coaster_df.manufacturer == 'RMC')
    gci_hybrid = (coaster_df.material_type == 'Hybrid') &\
        (coaster_df.manufacturer == 'GCI')
    gg_hybrid = (coaster_df.material_type == 'Hybrid') &\
        (coaster_df.manufacturer == 'Gravity Group')
    coaster_df.loc[(arrow_hybrid | rmc_hybrid), 'material_type'] = 'Wooden'
    coaster_df.loc[(gci_hybrid | gg_hybrid), 'material_type'] = 'Steel'
    coaster_df = coaster_df.rename(columns={'material_type': 'track_material'})
    coaster_df.track_material = coaster_df.track_material.astype('category')
    coaster_df = coaster_df[coaster_df.length > 0]
    coaster_df.reset_index(drop=True)
    return coaster_df


def wrangle_coasters() -> pd.DataFrame:
    coaster_df = acquire_coasters()
    coaster_df = prepare_coasters(coaster_df)
    return coaster_df


def tvt_split(df: pd.DataFrame,
              stratify: str = None,
              tv_split: float = .2,
              validate_split: int = .3):
    '''tvt_split takes a pandas DataFrame,
    a string specifying the variable to stratify over,
    as well as 2 floats where 0 < f < 1 and
    returns a train, validate, and test split of the DataFame,
    split by tv_split initially and validate_split thereafter. '''
    strat = df[stratify]
    train_validate, test = train_test_split(
        df, test_size=tv_split, random_state=69, stratify=strat)
    strat = train_validate[stratify]
    train, validate = train_test_split(
        train_validate, test_size=validate_split,
        random_state=69, stratify=strat)
    return train, validate, test


def man_groups(train: pd.DataFrame, validate: pd.DataFrame,
               test: pd.DataFrame) -> Tuple[pd.DataFrame,
                                            pd.DataFrame, pd.DataFrame]:
    grouping = train.groupby('manufacturer').mean().sort_values(
        by='length', ascending=False).index.to_list()
    train['man_group'] = -1
    validate['man_group'] = -1
    test['man_group'] = -1
    for i in range(5):
        train.loc[train.manufacturer.isin(
                grouping[i*5:(i+1)*5]), 'man_group'] = i
        validate.loc[validate.manufacturer.isin(
                grouping[i*5:(i+1)*5]), 'man_group'] = i
        test.loc[test.manufacturer.isin(
                grouping[i*5:(i+1)*5]), 'man_group'] = i
    return train, validate, test


