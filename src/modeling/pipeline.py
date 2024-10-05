from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


def create_numerical_transformer():
    """
    Create numerical transformer.

    Parameters:

    Returns:
    Pipeline(steps=[('imputer', SimpleImputer(missing_values=0, strategy='mean'))].
    """

    return Pipeline(steps=[
        ('imputer', SimpleImputer(missing_values=0, strategy='mean'))
    ])


def create_categorical_transformer():
    """
    Create categorical transformer.

    Parameters:

    Returns:
    Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))].
    """

    return Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

def create_geo_data_transformer(encoder):
    """
    Create geographical data transformer.

    Parameters:
    encoder.

    Returns:
    Pipeline(steps=[
        ('geo', encoder)
    ]).
    """
    return Pipeline(steps=[
        ('geo', encoder)
    ])


def create_preprocessor(numerical_transformer, categorical_transformer,
                        geo_categorical_transformer, numerical_columns_to_clean,
                        categorical_columns, geo_columns):
    """
    Create preprocessor.

    Parameters:
    numerical_transformer, categorical_transformer,
    geo_categorical_transformer, numerical_columns_to_clean,
    categorical_columns, geo_columns.

    Returns:
    ColumnTransformer(transformers=[
        ('numerical', numerical_transformer, numerical_columns_to_clean),
        ('categorical', categorical_transformer, categorical_columns),
        ('geo_categorical', geo_categorical_transformer, geo_columns),
    ]).
    """

    return ColumnTransformer(transformers=[
        ('numerical', numerical_transformer, numerical_columns_to_clean),
        ('categorical', categorical_transformer, categorical_columns),
        ('geo_categorical', geo_categorical_transformer, geo_columns),
    ])


def pipeline(preprocessor, model_name, model):
    """
    Pipeline.

    Parameters:
    preprocessor, model_name, model.

    Returns:
    Pipeline(steps=[('preprocessor', preprocessor),
                              (model_name, model)]).
    """

    return Pipeline(steps=[('preprocessor', preprocessor),
                              (model_name, model)])