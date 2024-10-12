# DATA CLEANING, TOKENIZATION, VECTORIZATION

import tensorflow as tf 
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class DataPreprocessor:
    """DataPreprocessor class for text data preprocessing"""

    def __init__(self, vocab_size=10000, max_len=100, padding_type='post', trunc_type='post', oov_token="<OOV>"):
        # Initialize preprocessing parameters
        self.vocab_size = vocab_size  # Maximum number of words to keep
        self.max_len = max_len  # Maximum length of sequences
        self.padding_type = padding_type  # Padding type (pre or post)
        self.trunc_type = trunc_type  # Truncation type (pre or post)
        self.oov_token = oov_token  # Out-of-vocabulary token
        # Initialize tokenizer with specified vocabulary size and OOV token
        self.tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
    
    def load_data(self, file_path):
        """
        Load data from CSV file.

        Args:
            file_path (str): Path to the CSV file.

        Returns:
            tuple: Reviews and Sentiments(positive or negative) from the dataset.
        """
        data = pd.read_csv(file_path)
        texts = data["review"].values
        labels = data['sentiment'].values
        
        return texts, labels
    
    def preprocess(self, texts):
        """
        Preprocess the input texts.

        Args:
            texts (list): List of input texts.

        Returns:
            numpy.ndarray: Padded sequences of tokenized texts.
        """
        # Fit tokenizer on texts and convert texts to sequences
        self.tokenizer.fit_on_texts(texts)
        sequences = self.tokenizer.texts_to_sequences(texts)
        # Pad sequences to ensure uniform length
        padded_sequences = pad_sequences(sequences, maxlen=self.max_len, 
                                         padding=self.padding_type, 
                                         truncating=self.trunc_type)
        
        return padded_sequences
    
    def split(self, texts, labels, test_size=0.2):
        """
        Split data into training and testing sets.

        Args:
            texts (list): List of preprocessed texts.
            labels (list): List of corresponding labels.
            test_size (float): Proportion of the dataset to include in the test split.

        Returns:
            tuple: Training and testing data splits.
        """
        return train_test_split(texts, labels, test_size=test_size)