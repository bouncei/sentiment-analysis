

import pandas as pd
import os
from src.data_preprocessing import DataPreprocessor
from src.config import Config

def preprocess_and_save():
    # Initialize the data preprocessor with configuration parameters
    preprocessor = DataPreprocessor(vocab_size=Config.VOCAB_SIZE, max_len=Config.MAX_LEN)
    
    # LOAD THE RAW DATA
    texts, labels = preprocessor.load_data(Config.DATA_PATH)
    print("Data loaded successfully.")
    

    #   PREPROCESS THE TEXT
    print("Cleaning and tokenizing texts...")
    sequences = preprocessor.preprocess(texts)
    print("Text preprocessing completed.")
    
    # SPLIT THE DATA
    X_train, X_val, y_train, y_val = preprocessor.split(sequences, labels, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE )
    
    
    # Convert to DataFrame for saving
    processed_train = pd.DataFrame(X_train)
    processed_train['label'] = y_train
    processed_val = pd.DataFrame(X_val)
    processed_val['label'] = y_val
    
    
    # Create processed data directory if it doesn't exist
    os.makedirs(os.path.dirname(Config.PROCESSED_DATA_PATH), exist_ok=True)
    
    # Save the processed data
    processed_train.to_csv(Config.PROCESSED_DATA_PATH.replace('.csv', '_train.csv'), index=False)
    processed_val.to_csv(Config.PROCESSED_DATA_PATH.replace('.csv', '_val.csv'), index=False)
    print(f"Processed training data saved to {Config.PROCESSED_DATA_PATH.replace('.csv', '_train.csv')}")
    print(f"Processed validation data saved to {Config.PROCESSED_DATA_PATH.replace('.csv', '_val.csv')}")
    
    # Save the tokenizer
    tokenizer_path = os.path.join(os.path.dirname(Config.MODEL_SAVE_PATH), 'tokenizer.pickle')
    preprocessor.save_tokenizer(tokenizer_path)


if __name__ == '__main__':
    preprocess_and_save()



    # Save processed data
    # processed_df = pd.DataFrame(sequences)
    # processed_df['label'] = labels
    # processed_df.to_csv(Config.PROCESSED_DATA_PATH, index=False)
    # print(f"Processed data saved to {Config.PROCESSED_DATA_PATH}")
