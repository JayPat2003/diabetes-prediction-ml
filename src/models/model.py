from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier

def train_classifier(X_train, y_train):
    classifier = HistGradientBoostingClassifier(random_state=42)
    classifier.fit(X_train, y_train)
    
    return classifier