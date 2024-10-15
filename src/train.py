# MODEL TRAINING PIPELINE

from data_preprocessing import DataPreprocessor
from model import SentimentAnalysisModel
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from src.config import Config
import pandas as pd
import os

def train_model():
    # Load the processed data
    train_path = Config.PROCESSED_DATA_PATH.replace('.csv', '_train.csv')
    val_path = Config.PROCESSED_DATA_PATH.replace('.csv', '_val.csv')

    print(f"Loading training data from {train_path}")
    train_df = pd.read_csv(train_path)
    X_train = train_df.drop('label', axis=1).values
    y_train = train_df['label'].values

    print(f"Loading validation data from {val_path}")
    val_df = pd.read_csv(val_path)
    X_val = val_df.drop('label', axis=1).values
    y_val = val_df['label'].values

    # Define the model
    model_obj = SentimentAnalysisModel(vocab_size=Config.VOCAB_SIZE, input_length=Config.MAX_LEN, embedding_dim=Config.EMBEDDING_DIM)
    model_obj.compile()
    model_obj.summary()

    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    tensorboard_callback = TensorBoard(log_dir=Config.LOG_DIR)

    # Train the model
    model = model_obj.get_model()
    history = model.fit(
        X_train, y_train,
        batch_size=Config.BATCH_SIZE,
        epochs=Config.EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, tensorboard_callback]
    )

    # Save the model
    os.makedirs(os.path.dirname(Config.MODEL_SAVE_PATH), exist_ok=True)
    model.save(Config.MODEL_SAVE_PATH)
    print(f"Model saved to {Config.MODEL_SAVE_PATH}")

if __name__ == '__main__':
    train_model()
