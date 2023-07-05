import xgboost as xgb

class XGBOOST:
    def __init__(self):
        self.model = xgb.XGBClassifier()

    def fit_xgb(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict_xgb(self, X_test):
        return self.model.predict(X_test)
    
    def return_model(self):
        return self.model