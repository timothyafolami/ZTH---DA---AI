import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def data_columns(df):
    """Return a list of column names in the DataFrame."""
    return list(df.columns)

def data_shape(df):
    """Return the shape of the DataFrame."""
    return df.shape

def data_info(df):
    """Return information about the DataFrame."""
    dtypes = ""
    for col in df.columns:
        dtypes += f"{col}: {df[col].dtype}\n"
    return dtypes

def sum_missing_values(df):
    """Return the sum of missing values in the DataFrame."""
    return df.isna().sum().sum()

def count_missing_numbers_per_column(df):
    """Return the count of missing values per column."""
    return df.isna().sum()

def sum_duplicates(df):
    """Return the sum of duplicates in the DataFrame."""
    return df.duplicated().sum()

def get_year_month_day(df, column_name):
    """Extract year, month, and day from a datetime column."""
    df[column_name] = pd.to_datetime(df[column_name])
    df['year'] = df[column_name].dt.year
    df['month'] = df[column_name].dt.month
    df['day'] = df[column_name].dt.day
    return df

def correlation_check(df, num):
    """Return a correlation matrix of the DataFrame."""
    return df[num].corr()

def numerical_columns(data):
    """Return a list of numerical columns in the DataFrame."""
    num = [col for col in data.columns if data[col].dtype in ['int64', 'float64']]
    return num

def categorical_columns(data):
    """Return a list of categorical columns in the DataFrame."""
    cat = [col for col in data.columns if data[col].dtype == 'object']
    return cat

def summary_stats(data):
    """Return the summary statistics of the DataFrame."""
    return data.describe()

def unique_values(data):
    """Return the unique values in the DataFrame."""
    return data.nunique()

def check_outliers(df, columns):
    # Calculate the mean and standard deviation of the data
    means = df[columns].mean()
    stds = df[columns].std()

    # Calculate the z-scores for each data point
    z_scores = (df[columns] - means) / stds

    # Identify outliers as those with a z-score greater than 3 or less than -3
    outliers = np.abs(z_scores) > 3

    # Calculate the percentage of outliers in each column
    percentages = (outliers.sum(axis=0) / len(df)) * 100

    return percentages