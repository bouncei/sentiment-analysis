import os

class Config:
    """This class contains the configuration details like hyperparameters, file paths, and other constants that you can easily change."""
    
    DATA_PATH = os.path.join('..', 'data', 'raw', 'IMDB_Dataset.csv')
    PROCESSED_DATA_PATH = os.path.join('..', 'data', 'processed', 'processed_data.csv')
    MODEL_SAVE_PATH = os.path.join('..', 'saved_models', 'sentiment_model.h5')
    VOCAB_SIZE = 10000
    MAX_LEN = 200  
    EMBEDDING_DIM = 128
    BATCH_SIZE = 64
    EPOCHS = 10
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    LOG_DIR = os.path.join('..', 'logs')  # TensorBoard logs
