from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier

def evaluate(train_X, train_y, val_X, val_y):
    #random forest with gini
    rf = RandomForestClassifier(criterion='entropy',n_estimators=200,max_depth=4,n_jobs=-1)

    rf.fit(train_X,train_y)

    rf_predict = rf.predict(val_X)

    f1_score(val_y, rf_predict, average="micro")

    return f1_score