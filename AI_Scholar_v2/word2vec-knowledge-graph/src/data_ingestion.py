import os
from PyPDF2 import PdfReader

def pdf_to_text(pdf_folder):
    texts = []
    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith('.pdf'):
            pdf_path = os.path.join(pdf_folder, pdf_file)
            reader = PdfReader(pdf_path)
            text = ''
            for page in reader.pages:
                text += page.extract_text()
            texts.append({'title': pdf_file, 'text': text})
    return texts

if __name__ == "__main__":
    pdf_folder = '//home/cmejo/arxiv-dataset/pdf'
    texts = pdf_to_text(pdf_folder)
    print(texts)
