

from src.data_preprocessing import DataPreprocessor
from src.config import Config
import pandas as pd

if __name__ == '__main__':
    # Load and preprocess the data
    preprocessor = DataPreprocessor(vocab_size=Config.VOCAB_SIZE, max_len=Config.MAX_LEN)
    texts, labels = preprocessor.load_data(Config.DATA_PATH)
    sequences = preprocessor.preprocess(texts)

    # Save processed data
    processed_df = pd.DataFrame(sequences)
    processed_df['label'] = labels
    processed_df.to_csv(Config.PROCESSED_DATA_PATH, index=False)
    print(f"Processed data saved to {Config.PROCESSED_DATA_PATH}")
