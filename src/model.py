# TENSORFLOW MODEL DEFINITION

import tensorflow as tf
from tensorflow.keras import layers

class SentimentAnalysisModel:
    """Sentiment model analysis model"""
    def __init__(self, vocab_size, embedding_dim=16, input_length=100):
        self.model = tf.keras.Sequential([
            layers.Embedding(vocab_size, embedding_dim, input_length=input_length),   # Convert input tokens to dense vectors
            layers.Bidirectional(layers.LSTM(64, return_sequence=True)),  # Processes the sequence in both directions, capturing context from past and future.
            layers.GlobalMaxPooling1D(), # Global max pooling to reduce sequence dimension
            layers.Dense(64, activation='relu'), # Dense layer with ReLU activation
            layers.Dense(1, activation='sigmoid') # For binary classification (positve/negative)
            
        ])
        
        
    def compile (self):
        """Compile the model with Adam optimizer, binary crossentropy loss, and accuracy metric"""
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
    def summary(self):
        """Print a summary of the model architecture"""
        self.model.summary()
        
    def get_model(self):
        """Return the constructed model"""
        return self.model
    
        
        
        