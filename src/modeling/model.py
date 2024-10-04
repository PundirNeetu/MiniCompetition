from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb



def create_random_forest_model():
    return RandomForestClassifier(criterion='entropy', n_estimators=200, max_depth=4, n_jobs=-1)

def create_xtreme_gradient_boosting_model():
    pass
    #return xgboost.XGBClassifier(*, objective='binary:logistic', **kwargs)