import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import tensorflow as tf

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

CLASS_NAMES = ['Iris Setosa', 'Iris Versicolor', 'Iris Virginica']
FEATURE_NAMES = [
    'Sepal Length (cm)',
    'Sepal Width (cm)',
    'Petal Length (cm)',
    'Petal Width (cm)'
]
FEATURE_KEYS = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

def load_iris_dataset():
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=FEATURE_KEYS)
    df['target'] = iris.target
    df['species'] = df['target'].map({0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'})
    return df, iris

def check_missing_values(df):
    null_counts = df.isnull().sum()
    null_percentages = (null_counts / len(df)) * 100
    missing_summary = pd.DataFrame({
        'Feature / Column': df.columns,
        'Missing Count': null_counts.values,
        'Missing Percentage (%)': null_percentages.values
    })
    return missing_summary

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
