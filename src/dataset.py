import pandas as pd
import os

        
def load_datasets(folder_path)-> pd.DataFrame:
    all_dataframes = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            try:
                df = pd.read_csv(file_path)
                if filename == 'test_values.csv':
                    all_dataframes['test_values'] = df
                elif filename=='train_labels.csv':
                    all_dataframes['train_labels'] = df
                elif  filename=='train_values.csv':
                    all_dataframes['train_values'] = df
                else:
                    continue
            except Exception as e:
                print(f"Error loading dataset {filename}: {e}")
    print("All datasets loaded successfully.")
    return all_dataframes



def preprocess_data(dataframe)-> pd.DataFrame:
    #Drop rows with any missing values
    df_cleaned = dataframe.dropna()
    print("Dropped rows with missing values.")
#    df_cleaned = remove_categorical_features(df_cleaned)
    return df_cleaned

def remove_categorical_features(dataframe)-> pd.DataFrame:
    return dataframe.select_dtypes(exclude=['object'])