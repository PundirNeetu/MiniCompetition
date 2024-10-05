import pandas as pd
import sys

sys.path.append('..')

train_values_path = '../data/train_values.csv'
train_labels_path = '../data/train_labels.csv'

def import_data(data_path):
    """
    Import data.

    Parameters:
    data_path

    Returns:
    df.
    """
    df = pd.read_csv(data_path)
    return df


raw_data_values = import_data(train_values_path)


# Missing values ('0') in 'age' and 'count_families'
# age:
# 0      26041
# count_families:
# 0     20862
def replace_zero_values_with_mean(column, df: pd.DataFrame):
    """
    Replace zero values with mean.

    Parameters:
    column, df: pd.DataFrame

    Returns:
    """
    mean_value = df[column].replace(0, pd.NA).mean()
    print(mean_value)
    df[column] = df[column].replace(0, mean_value)


def replace_zero_values_with_median(column, df: pd.DataFrame):
    """
    Replace zero values with median.

    Parameters:
    column, df: pd.DataFrame

    Returns:
    """
    median_value = df[column].replace(0, pd.NA).median()
    print(median_value)
    df[column] = df[column].replace(0, median_value)


def clean_numerical_data(columns, df: pd.DataFrame, replace_function):
    """
    Replace clean numerical data.

    Parameters:
    columns, df: pd.DataFrame, replace_function

    Returns:
    cleaned_data
    """
        
    cleaned_data = df.copy()
    for column in columns:
        replace_function(column, cleaned_data)
    return cleaned_data


columns_to_clean = ['age', 'count_families']

#  cleaned_numerical_data = clean_numerical_data(columns_to_clean, raw_data_values, replace_function=replace_zero_values_with_median)
