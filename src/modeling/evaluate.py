from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.modeling.pipeline import create_pipeline, create_preprocessor
from src.modeling.pipeline import create_categorical_transformer, create_numerical_transformer
from src.modeling.pipeline import create_random_forest_model
from src.modeling.model import create_xtreme_gradient_boosting_model


def create_rf_pipeline(categorical_columns, numerical_columns_to_clean) -> Pipeline:
    numerical_transformer = create_numerical_transformer()
    categorical_transformer = create_categorical_transformer()
    preprocessor = create_preprocessor(numerical_transformer, categorical_transformer, numerical_columns_to_clean, categorical_columns)
    rf_model = create_random_forest_model()
    return create_pipeline(preprocessor, model_name='randomforestclassifier', model=rf_model)


def create_xgb_pipeline(categorical_columns, numerical_columns_to_clean) -> Pipeline:
    numerical_transformer = create_numerical_transformer()
    categorical_transformer = create_categorical_transformer()
    preprocessor = create_preprocessor(numerical_transformer, categorical_transformer, numerical_columns_to_clean, categorical_columns)
    xgb_model = create_xtreme_gradient_boosting_model()
    return create_pipeline(preprocessor, model_name='xgbclassifier', model=xgb_model)


def evaluate(X_train, y_train, X_val, y_val, pipeline):
    #random forest with gini

    pipeline.fit(X_train, y_train)

    rf_predict = pipeline.predict(X_val)

    score = f1_score(y_val, rf_predict, average="micro")
    #print(f"F1-Score: {score}")

    return score