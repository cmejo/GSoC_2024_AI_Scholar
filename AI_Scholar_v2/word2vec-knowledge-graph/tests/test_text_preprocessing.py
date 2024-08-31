import unittest
from src.text_preprocessing import preprocess_text

class TestTextPreprocessing(unittest.TestCase):
    def test_preprocess_text(self):
        sample_text = "This is a sample text for preprocessing."
        tokens = preprocess_text(sample_text)
        self.assertIn('sample', tokens)
        self.assertNotIn('is', tokens)

if __name__ == "__main__":
    unittest.main()
