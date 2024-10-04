from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def column_remover(columns, columns_to_remove):
    columns_to_keep = [col for col in columns if col not in columns_to_remove]
    return ColumnTransformer(
        transformers=[
            ('keep', 'passthrough', columns_to_keep)
        ],
        remainder='drop'
    )

def create_numerical_transformer():
    return Pipeline(steps=[
        ('imputer', SimpleImputer(missing_values=0, strategy='mean'))
    ])


def create_categorical_transformer():
    return Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

def create_preprocessor(numerical_transformer, categorical_transformer, numerical_columns_to_clean, categorical_columns):

    return ColumnTransformer(transformers=[
        ('numerical', numerical_transformer, numerical_columns_to_clean),
        ('categorical', categorical_transformer, categorical_columns)
    ])


def create_random_forest_model():
    return RandomForestClassifier(criterion='entropy', n_estimators=200, max_depth=4, n_jobs=-1)


def create_pipeline(preprocessor, model_name, model):
    return Pipeline(steps=[('preprocessor', preprocessor),
                              (model_name, model)])