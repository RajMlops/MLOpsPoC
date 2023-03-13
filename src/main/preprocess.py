import sys
import pandas as pd
from src.utils.data_utils import *


def missingvalue(data):
    """ Identify and treat missing values """
    cols = data.select_dtypes('object').columns
    cols = cols.tolist()
    data['HomePlanet'].fillna(data['HomePlanet'].value_counts().index[0],
                              inplace=True)
    data['CryoSleep'].fillna(data['CryoSleep'].value_counts().index[0],
                             inplace=True)
    data['Destination'].fillna(data['Destination'].value_counts().index[0],
                               inplace=True)
    data['VIP'].fillna(data['VIP'].value_counts().index[0],
                       inplace=True)
    cols1 = data.select_dtypes('float64').columns
    cols1 = cols1.tolist()
    for i in cols1:
        data[i] = data[i].fillna(data[i].mean())
    return data


def onehotencoding(data):
    """ Use One-hot encoding """
    data = data.join(pd.get_dummies(data['HomePlanet'],
                     prefix='HomePlanet',
                     prefix_sep='_'))
    data = data.join(pd.get_dummies(data['CryoSleep'],
                     prefix='CryoSleep',
                     prefix_sep='_'))
    data = data.join(pd.get_dummies(data['Destination'],
                     prefix='Destination',
                     prefix_sep='_'))
    data = data.join(pd.get_dummies(data['VIP'],
                     prefix='VIP',
                     prefix_sep='_'))
    data.drop(['HomePlanet', 'CryoSleep', 'Destination', 'VIP'],
              axis=1,
              inplace=True)
    return data


def pre_processing(data):
    """ Preprocessing """
    data.drop(['PassengerId', 'Name', 'Cabin'], axis=1, inplace=True)
    data = missingvalue(data)
    data = onehotencoding(data)
    return data
