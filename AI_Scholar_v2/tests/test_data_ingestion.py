import unittest
from src.data_ingestion import pdf_to_text

class TestDataIngestion(unittest.TestCase):
    def test_pdf_to_text(self):
        pdf_folder = "tests/test_pdfs"
        texts = pdf_to_text(pdf_folder)
        self.assertTrue(len(texts) > 0)
        self.assertIn('text', texts[0])

if __name__ == "__main__":
    unittest.main()
