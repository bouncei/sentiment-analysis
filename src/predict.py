# REAL-TIME PREDICTIONS

from tensorflow.keras.models import load_model
from src.data_preprocessing import DataPreprocessor
from src.config import Config
import os

def predict_text(text):
    # Load the model
    model = load_model(Config.MODEL_SAVE_PATH)
    print(f"Loaded model from {Config.MODEL_SAVE_PATH}")

    # Initialize the data preprocessor and load the tokenizer
    preprocessor = DataPreprocessor(vocab_size=Config.VOCAB_SIZE, max_len=Config.MAX_LEN)
    tokenizer_path = os.path.join(os.path.dirname(Config.MODEL_SAVE_PATH), 'tokenizer.pickle')
    preprocessor.load_tokenizer(tokenizer_path)

    # Preprocess the input text
    text_seq = preprocessor.preprocess([text])  # Tokenize the input text
    print(f"Processed input text: {text_seq}")

    # Make prediction
    prediction = model.predict(text_seq)
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    
    return sentiment

# Example usage
if __name__ == '__main__':
    input_text = input("Enter text for sentiment prediction: ")
    sentiment = predict_text(input_text)
    print(f'Sentiment: {sentiment}')
