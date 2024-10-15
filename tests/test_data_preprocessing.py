# tests/test_data_preprocessing.py

import unittest
from src.data_preprocessing import DataPreprocessor
import os

class TestDataPreprocessor(unittest.TestCase):
    
    def setUp(self):
        self.preprocessor = DataPreprocessor(max_len=50)
        self.sample_texts = [
            "<html><body>I loved this movie! It's fantastic.</body></html>",
            "Terrible movie... It was a waste of time!!",
            "Average movie. Nothing special."
        ]
        self.sample_labels = [1, 0, 0]

    def test_clean_data(self):
        cleaned_texts = [self.preprocessor.clean_data(text) for text in self.sample_texts]
        expected_texts = [
            "loved movie fantastic",
            "terrible movie waste time",
            "average movie nothing special"
        ]
        self.assertEqual(cleaned_texts, expected_texts)

    def test_preprocess(self):
        cleaned_sequences = self.preprocessor.preprocess(self.sample_texts)
        self.assertEqual(cleaned_sequences.shape, (3, 50))  # Assuming max_len=50

    def test_split_data(self):
        sequences = self.preprocessor.preprocess(self.sample_texts)
        X_train, X_val, y_train, y_val = self.preprocessor.split(sequences, self.sample_labels, test_size=0.33)
        self.assertEqual(len(X_train), 2)
        self.assertEqual(len(X_val), 1)
        self.assertEqual(len(y_train), 2)
        self.assertEqual(len(y_val), 1)
        
    def test_save_and_load_tokenizer(self):
        self.preprocessor.preprocess(self.sample_texts)
        self.preprocessor.save_tokenizer(self.tokenizer_path)
        self.assertTrue(os.path.exists(self.tokenizer_path))

        # Create a new preprocessor instance and load the tokenizer
        new_preprocessor = DataPreprocessor(max_len=50)
        new_preprocessor.load_tokenizer(self.tokenizer_path)
        self.assertEqual(new_preprocessor.tokenizer.word_index, self.preprocessor.tokenizer.word_index)


if __name__ == '__main__':
    unittest.main()
