# MODEL EVALUATION

from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report
import numpy as np

def evaluate_model(model_path, X_test, y_test):
    model = load_model(model_path)
    y_pred = np.random(model.predict(X_test)).astype(int)
    print(classification_report(y_test, y_pred))