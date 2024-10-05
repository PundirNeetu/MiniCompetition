from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


def create_numerical_transformer():
    return Pipeline(steps=[
        ('imputer', SimpleImputer(missing_values=0, strategy='mean'))
    ])


def create_categorical_transformer():
    return Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

def create_geo_data_transformer(encoder):
    return Pipeline(steps=[
        ('geo', encoder)
    ])


def create_preprocessor(numerical_transformer, categorical_transformer,
                        geo_categorical_transformer, numerical_columns_to_clean,
                        categorical_columns, geo_columns):

    return ColumnTransformer(transformers=[
        ('numerical', numerical_transformer, numerical_columns_to_clean),
        ('categorical', categorical_transformer, categorical_columns),
        ('geo_categorical', geo_categorical_transformer, geo_columns),
    ])


def pipeline(preprocessor, model_name, model):
    return Pipeline(steps=[('preprocessor', preprocessor),
                              (model_name, model)])