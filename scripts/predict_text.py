# SCRIPT FOR RUNNING PREDICTIONS ON INPUT TEXT

from src.predict import predict_text

if __name__ == '__main__':
    model_path = '../saved_models/sentiment_model.h5'
    text = input("Enter text for sentiment prediction: ")
    sentiment = predict_text(model_path, text)
    print(f'Sentiment: {sentiment}')