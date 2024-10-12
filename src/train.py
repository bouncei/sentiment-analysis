# MODEL TRAINING PIPELINE
# 1. Data Preprocessing
# 2. Model Selection
# 3. Model Training
# 4. Model Evaluation
# 5. Model Deployment
# 6. Model Maintenance
# 7. Model Monitoring
# 8. Model Update

from data_preprocessing import DataPreprocessor
from model import SentimentAnalysisModel
from tensorflow.keras.callbacks import EarlyStopping

def train_model(data_path, vocal_size=10000, max_len=100, batch_size=32, epochs=5):
    """
    This function will handle the training process, including splitting the data and using the model.
    """
    
    # LOADS AND PREPORCESS THE DATA
    data_preprocessor = DataPreprocessor(vocal_size=vocal_size, max_len=max_len)
    texts, labels = data_preprocessor.load_data(data_path)
    sequences = data_preprocessor.preprocess(texts)
    X_train, X_val, y_train, y_val = data_preprocessor.split(sequences, labels)
    
    # DEFINE THE MODEL
    model_obj = SentimentAnalysisModel(vocal_size=vocal_size, input_length=max_len)
    model_obj.compile()
    model_obj.summary()
    
    # TRAIN THE MODEL
    model = model_obj.get_model()
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val), callbacks=[early_stopping])
    
    
    # SAVE THE MODEL 
    model.save('.../saved_models/sentiment_model.h5')
    print("Model training complete.")