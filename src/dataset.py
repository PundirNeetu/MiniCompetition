import pandas as pd
import os

class dataset(object):
    """
    A class to handle dataset.
    """
    def __init__(self, file_path):
        self.file_path = file_path
        
    def load_dataset(self, file_path):
        """
        Load a dataset from a CSV file.
        
        Parameters:
        file_path (str): The path to the CSV file.
        
        Returns:
        pd.DataFrame: The loaded DataFrame.
        """
        try:
            df = pd.read_csv(file_path)
            print(f"Successfully loaded dataset from {file_path}.")
            return df
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None

    def preprocess_data(self, dataframe):
        """
        Preprocess the DataFrame by filling or dropping missing values.
        
        Parameters:
        dataframe (pd.DataFrame): The DataFrame to preprocess.
        
        Returns:
        pd.DataFrame: The preprocessed DataFrame.
        """
        # Example: Drop rows with any missing values
        # Alternatively, you can fill missing values using:
        # dataframe.fillna(value=0, inplace=True)
        
        cleaned_df = dataframe.dropna()
        print("Dropped rows with missing values.")
        return cleaned_df
    def check_missing_values(self, dataframe):
        """
        check for missing value in ech columns of dataframe
        parameter:
        dataframe (pd.DataFrame): The data grame to check for missing value.

        returns:
        pd.series: A series with the count of missing values for each column
        """

        missing_values = dataframe.isnull().sum()
        print(missing_values)
        missing_values = missing_values[missing_values > 0]  
        return missing_values

    def set_dataset(self):
        self.data = self.load_dataset(self.file_path + "/" + 'train_values.csv')
        return self.check_missing_values(self.data)