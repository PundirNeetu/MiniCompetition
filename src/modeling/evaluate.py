import pandas as pd
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from src.modeling.pipeline import pipeline, create_preprocessor
from src.modeling.pipeline import create_categorical_transformer, create_numerical_transformer, create_geo_data_transformer
from src.modeling.model import create_random_forest_model, create_xtreme_gradient_boosting_model
from sklearn.model_selection import GridSearchCV
from src.modeling.model import create_xtreme_gradient_boosting_model
from category_encoders import TargetEncoder, CatBoostEncoder, HashingEncoder



def create_pipeline(categorical_columns, numerical_columns_to_clean, geo_columns, model_name) -> Pipeline:
    """
    Creates a pipeline.

    Parameters:
    categorical_columns
    numerical_columns_to_clean
    geo_columns
    model_name

    Returns:
    pipeline(preprocessor, model_name, model)
    """

    numerical_transformer = create_numerical_transformer()
    categorical_transformer = create_categorical_transformer()
    encoder = TargetEncoder()
    geo_transformer = create_geo_data_transformer(encoder)
    preprocessor = create_preprocessor(numerical_transformer, categorical_transformer,
                                       geo_transformer, numerical_columns_to_clean,
                                       categorical_columns, geo_columns)
    if model_name == 'rf':
        model = create_random_forest_model()
    elif model_name == 'xgb':
        model = create_xtreme_gradient_boosting_model()
    else:
        print('Not a valid model name')
        return
    return pipeline(preprocessor, model_name, model)


def create_xgb_pipeline(categorical_columns, numerical_columns_to_clean) -> Pipeline:
    """
    Creates xgb pipeline.

    Parameters:
    categorical_columns
    numerical_columns_to_clean

    Returns:
    create_pipeline(preprocessor, model_name='xgbclassifier', model=xgb_model)
    """
    numerical_transformer = create_numerical_transformer()
    categorical_transformer = create_categorical_transformer()
    preprocessor = create_preprocessor(numerical_transformer, categorical_transformer, numerical_columns_to_clean, categorical_columns)
    xgb_model = create_xtreme_gradient_boosting_model()
    return create_pipeline(preprocessor, model_name='xgbclassifier', model=xgb_model)


def evaluate(X_train, y_train, X_val, y_val, pipeline):
    """
    Evaluate the model performance.

    Parameters:
    X_train
    y_train
    X_val
    y_val
    pipeline

    Returns:
    score
    """
    #random forest with gini
    pipeline.fit(X_train, y_train)
    rf_predict = pipeline.predict(X_val)
    score = f1_score(y_val, rf_predict, average="micro")
    return score

def perform_crossvalidation(model, output_file_number, pipeline, X, y, cv=5):
    """
    Perform crossvalidation.

    Parameters:
    model
    output_file_number
    pipeline
    X
    y
    cv

    Returns:
    """

    if model == 'rf':
        param_grid = {
            'randomforestclassifier__n_estimators': [100],
            'randomforestclassifier__max_depth': [5]
        }
    elif model == 'xgb':
        param_grid = {
            'xgbclassifier__n_estimators': [400, 600],
            'xgbclassifier__learning_rate': [0.07, 0.1, 0.13],
            'xgbclassifier__max_depth': [3, 5, 7]
        }
    else:
        print('Not a valid model name')
        return

    grid_search = GridSearchCV(pipeline, param_grid, cv=cv, return_train_score=True)
    grid_search.fit(X, y)

    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validation score:", grid_search.best_score_)

    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df.to_csv(f'src/modeling/crossvalidation_output_{model}_{output_file_number}.csv')
