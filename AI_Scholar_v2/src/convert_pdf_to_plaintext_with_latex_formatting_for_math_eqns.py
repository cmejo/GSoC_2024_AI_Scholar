import os
import PyPDF2
import re

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a single PDF file.
    """
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text

def format_math_equations(text):
    """
    Applies LaTeX formatting to detected math equations.
    """
    # Example: Simple pattern to detect math equations
    # This is a basic example; real-world PDFs may require more complex regex patterns
    inline_math_pattern = re.compile(r'\$(.*?)\$')
    display_math_pattern = re.compile(r'\$\$(.*?)\$\$', re.DOTALL)

    # Convert detected inline math to LaTeX
    text = inline_math_pattern.sub(r'\\(\1\\)', text)
    
    # Convert detected display math to LaTeX
    text = display_math_pattern.sub(r'\\[\1\\]', text)

    return text

def save_text_to_file(text, output_path):
    """
    Saves extracted text to a .txt file.
    """
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(text)

def convert_pdfs_to_text(pdf_folder, output_folder):
    """
    Converts all PDF files in a folder to plain text files.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_folder, pdf_file)
        text = extract_text_from_pdf(pdf_path)
        
        # Format text with LaTeX for math equations
        text = format_math_equations(text)
        
        # Save to a .txt file
        output_file = os.path.splitext(pdf_file)[0] + '.txt'
        output_path = os.path.join(output_folder, output_file)
        save_text_to_file(text, output_path)

        print(f"Converted {pdf_file} to {output_file}")

# Example usage
pdf_folder = '/home/cmejo/arxiv-dataset/pdf'
output_folder = '/home/cmejo/arxiv-dataset/plain-text-converted'
convert_pdfs_to_text(pdf_folder, output_folder)
