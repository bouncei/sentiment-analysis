# UNIT TEST FOR DATA PREPROCESSING


import unittest
from src.data_preprocessing import DataPreprocessor

class TestDataPreprocessor(unittest.TestCase):
    """
    Unit test for data preprocessing
    """
    def test_preprocess(self):
        data_preprocessor = DataPreprocessor()
        sample_texts = [
            "I love this movie",
            "I hate this movie",
            "The food was absolutely delicious!",
            "This book is boring and poorly written.",
            "I can't believe how expensive these tickets are.",
            "The customer service was outstanding.",
            "The weather today is perfect for a picnic.",
            "That concert was the worst experience of my life.",
            "I'm feeling a bit under the weather today.",
            "The new software update fixed all the bugs.",
            "This product doesn't work as advertised.",
            "I'm so excited for my upcoming vacation!",
            "The traffic in this city is getting worse every day.",
            "She's always been a great friend to me.",
            "I don't understand why this movie got such good reviews.",
            "The hotel room was clean and comfortable.",
            "This class is way too difficult for beginners."
            ] 
        sequences = data_preprocessor.preprocess(sample_texts)
        self.assertEqual(len(sequences[0]), data_preprocessor.max_len)
        
        

if __name__ == '__main__':
    unittest.main()