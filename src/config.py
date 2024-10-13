# src/config.py

class Config:
    """This class contains the configuration details like hyperparameters, file paths, and other constants that you can easily change."""
    
    DATA_PATH = '../data/raw/IMDB_Dataset.csv'
    PROCESSED_DATA_PATH = '../data/processed/processed_data.csv'
    MODEL_SAVE_PATH = '../saved_models/sentiment_model.h5'
    VOCAB_SIZE = 10000
    MAX_LEN = 100
    EMBEDDING_DIM = 16
    BATCH_SIZE = 32
    EPOCHS = 5
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
