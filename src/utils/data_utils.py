import pandas as pd
import sys
import numpy as np
from sklearn.model_selection import train_test_split


def train_load_data() -> pd.DataFrame:
    train_df = pd.read_csv('data/train.csv')
    return train_df


def test_load_data() -> pd.DataFrame:
    test_df = pd.read_csv('data/test.csv')
    sub = pd.DataFrame(test_df['PassengerId'])
    return test_df


def stratified_split(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x,
                                                        y,
                                                        random_state=5,
                                                        stratify=y,
                                                        test_size=0.40)
    return x_train, x_test, y_train, y_test
