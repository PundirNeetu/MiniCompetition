import pandas as pd
import numpy as np
#import sys
from sklearn.pipeline import Pipeline

#sys.path.append('..')


def create_prediction(test_data: pd.DataFrame, pipeline:Pipeline) -> np.array:
    """
    Create prediction.

    Parameters:
    test_data: pd.DataFrame, pipeline:Pipeline

    Returns:
    pipeline.predict(test_data).
    """
    return pipeline.predict(test_data)


def create_output(test_data: pd.DataFrame, prediction: np.array, output_file_number: str):
    """
    Create output in the submission format.

    Parameters:
    test_data: pd.DataFrame, prediction: np.array, output_file_number: str.

    Returns:
    """
        
    output = pd.DataFrame()
    output['building_id'] = test_data['building_id']
    output['damage_grade'] = prediction
    #output['damage_grade'] = prediction + 1 # we add 1 here, becaue it was subtracted in main
    output.to_csv(f'data/output_{output_file_number}.csv', index=False)