import pandas as pd
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.modeling.pipeline import create_pipeline, create_preprocessor
from src.modeling.pipeline import create_categorical_transformer, create_numerical_transformer
from src.modeling.pipeline import create_random_forest_model
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import GridSearchCV


def create_rf_pipeline(categorical_columns, numerical_columns_to_clean, model) -> Pipeline:
    numerical_transformer = create_numerical_transformer()
    categorical_transformer = create_categorical_transformer()
    preprocessor = create_preprocessor(numerical_transformer, categorical_transformer, numerical_columns_to_clean, categorical_columns)
    rf_model = create_random_forest_model()
    return create_pipeline(preprocessor, model_name='randomforestclassifier', model=rf_model)


def evaluate(X_train, y_train, X_val, y_val, pipeline):
    #random forest with gini

    pipeline.fit(X_train, y_train)

    rf_predict = pipeline.predict(X_val)

    score = f1_score(y_val, rf_predict, average="micro")
    print(f"F1-Score: {score}")

    return score


def perform_crossvalidation(pipeline, X, y, cv=5):
    param_grid = {
        'randomforestclassifier__n_estimators': [200, 300],
        'randomforestclassifier__max_depth': [5, 10]
    }

    # Set up GridSearchCV with cross-validation
    grid_search = GridSearchCV(pipeline, param_grid, cv=cv, return_train_score=True)

    # Fit GridSearchCV to the data
    grid_search.fit(X, y)

    # Print best parameters and best cross-validation score
    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validation score:", grid_search.best_score_)

    # Convert cv_results_ to a DataFrame for easy viewing
    results_df = pd.DataFrame(grid_search.cv_results_)

    # Überprüfe, welche Spalten in cv_results_ verfügbar sind
    print("\nAvailable columns in cv_results_:")
    print(results_df.columns)

    # Print a summary of the results, including parameters and mean validation scores
    columns_to_print = ['params', 'mean_test_score', 'std_test_score', 'rank_test_score']

    # Füge 'mean_train_score' und 'std_train_score' nur hinzu, wenn sie vorhanden sind
    if 'mean_train_score' in results_df.columns and 'std_train_score' in results_df.columns:
        columns_to_print.extend(['mean_train_score', 'std_train_score'])

    print("\nDetailed cross-validation results:")
    print(results_df[columns_to_print])