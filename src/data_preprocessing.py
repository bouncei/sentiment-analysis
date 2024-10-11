# DATA CLEANING, TOKENIZATION, VECTORIZATION
import tensorflow as tf 
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


class DataPreprocessor:
    def __init__(self, vocab_size=10000, max_len=100, padding_type='post',trunc_type='post', oov_token="<OOV>" ):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.padding_type = padding_type
        self.trunc_type = trunc_type
        self.oov_token = oov_token
        self.tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
        
    
    def load_data(self, file_path):
        # LOAD CSV OR JSON DATASET
        data = pd.read_csv(file_path)
        texts = data["text"].values
        labels = data['label'].values
        
        return texts, labels
    
    def preprocess(self, texts):
        # TOKENIZE TEXTS AND PAD SEQUENCES
        self.tokenizer.fit_on_texts(texts)
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_len, padding=self.padding_type, truncating=self.trunc_type)
        
        return pad_sequences
    
    def split(self, texts, labels, test_size=0.2):
        # SPLIT DATA INTO TRAIN AND TEST SETS
        return train_test_split(texts, labels, test_size=test_size)
    
    


