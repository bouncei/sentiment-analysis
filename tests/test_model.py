# UNIT TEST FOR MODEL

import unittest
from src.model import SentimentAnalysisModel
from src.config import Config

class TestModel(unittest.TestCase):
    """
    Test to ensure that the model is created correctly.
    """
    
    def test_model_creation(self):
        model_obj = SentimentAnalysisModel(vocab_size=Config.VOCAB_SIZE, input_length=Config.MAX_LEN)
        model_obj.compile()
        model = model_obj.get_model()
        
        self.assertEqual(model.input_shape, (None, Config.MAX_LEN), "Model input shape mismatch")
        self.assertEqual(model.output_shape, (None, 1), "Model output shape mismatch")


if __name__ == '__main__':
    unittest.main()