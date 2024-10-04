import pandas as pd
import numpy as np
import sys
from sklearn.pipeline import Pipeline


sys.path.append('..')
# TEST_DATA_PATH = '../data/test_values.csv'


# def read_test_data(test_data_path: str) -> pd.DataFrame:
#    return pd.read_csv(test_data_path)


def create_prediction(test_data: pd.DataFrame, pipeline:Pipeline) -> np.array:
    return pipeline.predict(test_data)


def create_output(test_data: pd.DataFrame, prediction: np.array, output_file_number: str):
    output = pd.DataFrame()
    output['building_id'] = test_data['building_id']
    output['damage_grade'] = prediction
    output.to_csv(f'data/output_{output_file_number}.csv', index=False)