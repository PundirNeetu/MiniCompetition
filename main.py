import pandas as pd
from src.dataset import load_datasets, preprocess_data
from src.features import create_features
from src.config import DATA_FOLDER
from src.modeling.evaluate import evaluate, create_rf_pipeline
from src.modeling.split import split
from src.modeling.predict import create_prediction, create_output

def main()-> pd.DataFrame:
    # Load all datasets from the data folder
    all_dataframes = load_datasets(DATA_FOLDER)
    if len(all_dataframes)<3:
        print('Input file is missing, pls fix and re-run')
        return  # Exit if no datasets could be loaded
    
    # Pre-processing the dataset
    df_all_preprocessing = {}
    for key,value in all_dataframes.items():
        print(f"Preprocessing {key}:")
        df_all_preprocessing[key] = preprocess_data(value)

    # Assuming you want to evaluate a specific DataFrame, e.g., 'train_data'
    df = pd.merge(df_all_preprocessing['train_values'], df_all_preprocessing['train_labels'], on='building_id', how='inner')
    print("Merged DataFrame shape:", df.shape)
    X = df.drop(columns=['damage_grade'])  
    y = df['damage_grade']  

    # Use the split function to get training and validation sets
    train_X, val_X, train_y, val_y = split(X, y)

    categorical_columns = all_dataframes['train_values'].select_dtypes(include='object').columns
    numerical_columns_to_clean = ['age']

    pipeline = create_rf_pipeline(categorical_columns, numerical_columns_to_clean)

    # Call the evaluate function to get the F1 score
    f1 = evaluate(train_X, train_y, val_X, val_y, pipeline)

    # Print the F1 score
    print("F1 Score:", f1)

    prediction = create_prediction(test_data=all_dataframes['test_values'], pipeline=pipeline)
    create_output(test_data=all_dataframes['test_values'], prediction=prediction, output_file_number='01')

if __name__ == "__main__":
    main()