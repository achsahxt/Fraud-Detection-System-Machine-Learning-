from xgboost import XGBClassifier

def train_model(X_train, y_train):
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=50,
        eval_metric="logloss"
    )
    model.fit(X_train, y_train)
    return model
