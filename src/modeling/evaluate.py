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


def perform_crossvalidation_RF(output_file_number, model_used, pipeline, X, y, cv=5):
    param_grid = {
        'randomforestclassifier__n_estimators': [50, 100],
        'randomforestclassifier__max_depth': [5, 7]
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=cv, return_train_score=True)

    grid_search.fit(X, y)

    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validation score:", grid_search.best_score_)

    results_df = pd.DataFrame(grid_search.cv_results_)

    results_df.to_csv(f'src/modeling/crossvalidation_output_RF_{output_file_number}.csv')