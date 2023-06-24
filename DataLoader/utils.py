import numpy as np
np.random.seed(1337)  # for reproducibility
import pandas as pd
import os
from typing import List, Tuple, Generator, Iterator
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import sys
from tabulate import tabulate
import csv
import time
from natsort import natsorted
import pickle
import multiprocessing as mp
from os import path

from sklearn import preprocessing
from sklearn import base
from sklearn.preprocessing import (StandardScaler, OrdinalEncoder, LabelEncoder, MinMaxScaler, OneHotEncoder)
from sklearn.model_selection import train_test_split


TupleOrList = tuple([Tuple, List])


def Interpolate_with_gaussian_noise(data: pd.Series) -> pd.Series:
  """Couldn't find a proper name. Very slow ..."""
  DTYPE = np.float32

  series = data.astype(DTYPE)
  values = series.tolist()
  processed = []

  series_size = len(values)

  prev_rssi = 0
  prev_seq = -1
  for seq, rssi in enumerate(values):
    if not np.isnan(rssi):
        avg_rssi = np.mean([prev_rssi, rssi])
        std_rssi = np.std([prev_rssi, rssi])
        std_rssi = std_rssi if std_rssi > 0 else np.nextafter(DTYPE(0), DTYPE(1))
        diff = seq - prev_seq - 1

        processed.extend(np.random.normal(avg_rssi, std_rssi, size=diff))
        processed.append(rssi)
        prev_seq, prev_rssi = seq, rssi

  avg_rssi = np.mean([prev_rssi, 0.])
  std_rssi = np.std([prev_rssi, 0.])
  diff = series_size - prev_seq - 1
  processed.extend(np.random.normal(avg_rssi, std_rssi, size=diff))

  series = pd.Series(data=processed, index=data.index, dtype=DTYPE)
  return series


def Interpolate_with_constant(data: pd.Series, constant: int = 0) -> pd.Series:
  """Interpolate missing values with constant value."""
  return data.fillna(value=constant)


class CustomInterpolation(base.BaseEstimator, base.TransformerMixin):
  """Custom interpolation function to be used in"""

  STRATEGIES_ALL = ['none', 'gaussian', 'constant']

  def __init__(self, source: str, strategy: str = 'constant', constant: float = 0, target=None):
    if strategy not in self.STRATEGIES_ALL:
      raise ValueError(f'"{strategy}" is not available strategy')

    self.strategy = strategy
    self.constant = constant

    self.source = source
    self.target = source if target is None else target

  def with_constant(self, data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    df[self.target] = df[self.source].fillna(value=self.constant)
    return df

  def with_gaussian(self, data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    df[self.target] = Interpolate_with_gaussian_noise(df[self.source])
    return df

  def with_none(self, data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    src = [self.source] if isinstance(self.source, [str]) else self.source
    df = df.dropna(subset=src)
    return df

  def do_interpolation(self, X: pd.DataFrame) -> pd.DataFrame:
    if self.strategy == 'constant':
      return self.with_constant(X)

    if self.strategy == 'gaussian':
      return self.with_gaussian(X)

    if self.strategy == 'none':
      return self.with_none(X)

    raise ValueError(f'"{self.strategy}" is not available strategy')

  def fit(self, X: pd.DataFrame, y=None):
    return self

  def transform(self, X, y='deprecated', copy=True):
    if isinstance(X, (List, Tuple,)):
      with mp.Pool(processes=2) as p:
          return p.map(self.do_interpolation, X)

    return self.do_interpolation(X)


class CustomSplitter(base.BaseEstimator, base.TransformerMixin):
  def __init__(self, X: TupleOrList = None, y: str = 'class', drop: TupleOrList = None):
    self.X = X
    self.y = y
    self.drop = drop

  def fit(self, X: pd.DataFrame, y=None):
    return self

  def transform(self, df: pd.DataFrame, y='deprecated', copy=True):
    df = df.copy() if copy else df
    if self.drop:
      df.drop(labels=self.drop, axis=1, inplace=True)

    if self.X:
      return df[self.X], df[self.y].ravel()

    return df.drop(self.y), df[self.y].ravel()


class CustomMerger(base.BaseEstimator, base.TransformerMixin):
  """Merge List of DataFrames"""

  def __init__(self):
    pass

  def fit(self, X: pd.DataFrame, y=None):
    return self

  def transform(self, X: pd.DataFrame, y='deprecated', copy=True):
    if isinstance(X, TupleOrList):
        return pd.concat(X, ignore_index=True).reset_index(drop=True)

    return X


class SyntheticFeatures(base.BaseEstimator, base.TransformerMixin):
  """Rolling window for mean & std features."""

  def __init__(self, source: str, window_size: int = 10, target=None):
    self.source = source
    self.target = source if target is None else target

    if not isinstance(window_size, int) or not window_size > 0:
      raise ValueError(f'Window should be positive integer. Got `{window_size}` instead.')

    self.window = window_size

  def fit(self, X, y=None):
    return self

  def do_synthetics(self, data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    df[f'{self.target}_mean'] = df[self.source].rolling(self.window).mean()
    df[f'{self.target}_std'] = df[self.source].rolling(self.window).std()
    df[f'{self.target}_median'] = df[self.source].rolling(self.window).median()
    return df

  def transform(self, X: pd.DataFrame, y='deprecated', copy=True):
    if isinstance(X, (List, Tuple,)):
      with mp.Pool(processes=2) as p:
          return p.map(self.do_synthetics, X)

    return self.do_synthetics(X)


def Poly_features(df: pd.DataFrame, include: List[str], degree: int, include_bias=False, *args,
                **kwargs) -> pd.DataFrame:
  """The `PolynomialFeatures` from sklern drops/loses information about column names from pandas, which is not very convinient.
  This is a workaround for this behaviour to preserve names.
  """
  X, excluded = df.loc[:, include], df.drop(include, axis=1)
  poly = preprocessing.PolynomialFeatures(degree=degree, include_bias=include_bias, *args, **kwargs).fit(X)

  # Next line converts back to pandas, while preserving column names
  X = pd.DataFrame(poly.transform(X), columns=poly.get_feature_names(X.columns), index=X.index)

  data = pd.concat([X, excluded], axis=1, )
  data = data.reset_index(drop=True)

  # Transform column names. Ex. 'rssi rssi_avg' -> 'rssi*rssi_avg'
  data = data.rename(lambda cname: cname.replace(' ', '*'), axis='columns')

  return data