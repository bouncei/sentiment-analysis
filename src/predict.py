# REAL-TIME PREDICTIONS

from tensorflow.keras.models import load_model
from data_preprocessing import DataPreprocessor

def predict_text(model_path, text):
    model = load_model(model_path)
    data_preprocessor = DataPreprocessor()
    
    text_seq = data_preprocessor.preprocess([text]) # Tokenize the input text
    prediction = model.predict(text_seq)
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    
    return sentiment


    