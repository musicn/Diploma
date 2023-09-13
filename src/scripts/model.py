import xgboost as xgb
from sklearn.linear_model import LogisticRegression

class XGBOOST:
    def __init__(self):
        self.model = xgb.XGBClassifier()

    def fit_xgb(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict_xgb(self, X_test):
        return self.model.predict(X_test)
    
    def return_model(self):
        return self.model
    
class LOG_REG:
    def __init__(self):
        self.model = LogisticRegression(multi_class='ovr', max_iter=1000, random_state=42)

    def fit_log_reg(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict_log_reg(self, X_test):
        return self.model.predict(X_test)
    
    def return_model(self):
        return self.model