# DATA CLEANING, TOKENIZATION, VECTORIZATION

import tensorflow as tf 
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
import nltk
import pickle

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
        self.stop_words = set(stopwords.words('english'))

    
    def load_data(self, file_path):
        """
        Load data from CSV file.

        Args:
            file_path (str): Path to the CSV file.

        Returns:
            tuple: texts and labels(positive or negative) from the dataset.
        """
        data = pd.read_csv(file_path)
        texts = data["review"].values
        labels = data['sentiment'].values
        
        return texts, labels
    
    def clean_data(self, text):
        """
        Clean the input text by removing HTML tags, special characters, and stopwords.
        
        Args:
            text (str): text input.
        
        Return:
            str: cleaned text.
        
        """
        
        # REMOVE HTML TAGS
        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text()
        
        # REMOVE NON-LETTERS (replace non-letters with "")
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # REMOVE EXTRA WHITESPACE (replace extra white space with single whitespace)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # REMOVE STOPWORDS
        tokens = text.split()
        tokens = [word for word in tokens if word not in self.stop_words]
        cleaned_text = ' '.join(tokens)
        
        return cleaned_text
        
        
    
    def preprocess(self, texts):
        """
        Preprocess the input texts.

        Args:
            texts (list): List of input texts.

        Returns:
            numpy.ndarray: Padded sequences of tokenized texts.
        """
        
        # CLEAN THE TEXTS
        cleaned_texts = [self.clean_data(text) for text in texts]
        
        # Fit tokenizer on texts and convert texts to sequences
        self.tokenizer.fit_on_texts(cleaned_texts)
        sequences = self.tokenizer.texts_to_sequences(cleaned_texts)
        # Pad sequences to ensure uniform length
        padded_sequences = pad_sequences(sequences, maxlen=self.max_len, 
                                         padding=self.padding_type, 
                                         truncating=self.trunc_type)
        
        return padded_sequences
    
    def split(self, sequences, labels, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets.

        Args:
            texts (list): List of preprocessed texts.
            labels (list): List of corresponding labels.
            test_size (float): Proportion of the dataset to include in the test split.

        Returns:
            tuple: Training and testing data splits.
        """
        return train_test_split(sequences, labels, test_size=test_size, random_state=random_state)
    
    def save_tokenizer(self, save_path):
        """
        Save the tokenizer to a file.
        """
        with open(save_path, 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Tokenizer saved to {save_path}")

    def load_tokenizer(self, load_path):
        """
        Load the tokenizer from a file.
        """
        with open(load_path, 'rb') as handle:
            self.tokenizer = pickle.load(handle)
        print(f"Tokenizer loaded from {load_path}")