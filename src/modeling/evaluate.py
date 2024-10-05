import pandas as pd
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
import xgboost as xgb
from src.modeling.pipeline import pipeline, create_preprocessor
from src.modeling.pipeline import create_categorical_transformer, create_numerical_transformer, create_geo_data_transformer
from sklearn.model_selection import GridSearchCV
from category_encoders import TargetEncoder, CatBoostEncoder, HashingEncoder



def create_pipeline(categorical_columns, numerical_columns_to_clean, geo_columns) -> Pipeline:

    numerical_transformer = create_numerical_transformer()
    categorical_transformer = create_categorical_transformer()
    encoder = FrequencyEncoder()
    geo_transformer = create_geo_data_transformer(encoder)
    preprocessor = create_preprocessor(numerical_transformer=numerical_transformer,
                                       categorical_transformer=categorical_transformer,
                                       geo_categorical_transformer=geo_transformer,
                                       numerical_columns_to_clean=numerical_columns_to_clean,
                                       categorical_columns=categorical_columns,
                                       geo_columns=geo_columns)
    model = xgb.XGBClassifier(n_estimators = 400, learning_rate = 0.1, max_depth = 5)
    return pipeline(preprocessor, model_name='xgbclassifier', model=model)


def evaluate(X_train, y_train, X_val, y_val, pipeline):
    #random forest with gini
    pipeline.fit(X_train, y_train)
    rf_predict = pipeline.predict(X_val)
    score = f1_score(y_val, rf_predict, average="micro")
    return score

def perform_crossvalidation(cv_output_file_number, pipeline, X, y, cv=5):
    param_grid = {
        'xgbclassifier__n_estimators': [400],
        'xgbclassifier__learning_rate': [0.1],
        'xgbclassifier__max_depth': [5]
    }
    grid_search = GridSearchCV(pipeline, param_grid, cv=cv, return_train_score=True)
    grid_search.fit(X, y)

    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validation score:", grid_search.best_score_)

    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df.to_csv(f'src/modeling/crossvalidation_output_{cv_output_file_number}.csv')

    best_model = grid_search.best_estimator_
    return best_model


from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.freq_map = {}

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            for col in X.columns:
                self.freq_map[col] = X[col].value_counts(normalize=True)
        else:
            raise ValueError("Input should be a pandas DataFrame.")
        return self

    def transform(self, X):
        X_encoded = X.copy()
        for col in X_encoded.columns:
            X_encoded[col] = X_encoded[col].map(self.freq_map[col])
        return X_encoded