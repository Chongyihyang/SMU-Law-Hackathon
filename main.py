from pdfReader import PDFProcessor
from easyOCR import EasyOCRWithRiskScore
import os

class main:

    def run_code(file_path, output_dir="pdf_output"):
    
        if file_path.lower().endswith('.pdf'):        
            processor = PDFProcessor()
            processor.process_pdf(file_path, output_dir)

        elif file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            ocr = EasyOCRWithRiskScore()
            ocr.extract_text_with_confidence(file_path, output=True, output_dir=output_dir)

        elif file_path.lower().endswith('.txt'):
            txt_path = os.path.join(output_dir, f"{file_path}__100__.txt")
            stuff = ""
            with open(file_path, 'r', encoding='utf-8') as txt_file:
                stuff = txt_file.read()
            with open(txt_path, 'w', encoding='utf-8') as txt_file:
                txt_file.write(stuff.strip())

        else:
            print(f"Unsupported file type: {file_path}. Please provide a PDF, image, or text file.")


main.run_code("test.png")