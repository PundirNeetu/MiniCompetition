import pandas as pd
from src.dataset import load_datasets, preprocess_data
from src.features import create_features
from src.config import DATA_FOLDER
from src.modeling.evaluate import evaluate, create_pipeline
from src.modeling.evaluate import perform_crossvalidation
from src.modeling.split import split
from src.modeling.predict import create_prediction, create_output
from sklearn.preprocessing import LabelEncoder
from src.plots import plot_numerical_distributions, plot_categorical_distributions, plot_heatmap_with_target, plot_target_distribution




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
    X = df.drop(columns=['damage_grade', 'building_id'])
    y = df['damage_grade']
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    geo_columns = ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id' ]
    X[geo_columns] = X[geo_columns].astype(str)

    # Plotting
    print(plot_numerical_distributions(X))
    plot_categorical_distributions(X)
    plot_heatmap_with_target(X, df_all_preprocessing['train_labels'])
    plot_target_distribution(y)


    # Use the split function to get training and validation sets
    train_X, val_X, train_y, val_y = split(X, y)

    categorical_columns = all_dataframes['train_values'].select_dtypes(include='object').columns
    numerical_columns_to_clean = ['age']


    # Create a pipeline and optimize parameters with grid search and cross-validation
    initial_pipeline = create_pipeline(categorical_columns, numerical_columns_to_clean, geo_columns)
    best_pipeline = perform_crossvalidation(cv_output_file_number=3, pipeline=initial_pipeline, X=X, y=y)

    # Call the evaluate function to get the F1 score
    f1 = evaluate(train_X, train_y, val_X, val_y, best_pipeline)

    # Print the F1 score
    print("F1 Score:", f1)


    prediction = create_prediction(test_data=all_dataframes['test_values'], pipeline=best_pipeline)
    prediction_decoded = label_encoder.inverse_transform(prediction)
    prediction_df = pd.DataFrame(prediction)
    print(prediction_df.describe())
    print(prediction_df.value_counts())

    create_output(test_data=all_dataframes['test_values'], prediction=prediction_decoded, output_file_number='XGB_06')


if __name__ == "__main__":
    main()