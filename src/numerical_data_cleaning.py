import pandas as pd
import sys
sys.path.append('..')

train_values_path = '../data/train_values.csv'

def import_data(data_path):
    df = pd.read_csv(data_path)
    return df


raw_data_values = import_data(train_values_path)

# Missing values ('0') in age and count_families
# age:
# 0      26041
# count_families:
# 0     20862
def replace_zero_values(column, df: pd.DataFrame):
    mean_value = df[column].replace(0, pd.NA).mean()
    print(mean_value)
    df[column] = df[column].replace(0, mean_value)

def clean_numerical_data(columns, df: pd.DataFrame):
    cleaned_data = df.copy()
    for column in columns:
        replace_zero_values(column, cleaned_data)
    return cleaned_data


columns_to_clean = ['age', 'count_families']

cleaned_numerical_data = clean_numerical_data(columns_to_clean, raw_data_values)