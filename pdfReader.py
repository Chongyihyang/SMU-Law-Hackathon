import os
import PyPDF2
from pdf2image import convert_from_path
from easyOCR import EasyOCRWithRiskScore

class PDFProcessor:
    def __init__(self):
        self.ocr = EasyOCRWithRiskScore()

    def is_pdf_scanned(pdf_path):
        """
        Checks if a PDF contains extractable text.
        Returns True if scanned (no text), False if it has text.
        """
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                # Check every page for text
                for page in reader.pages:
                    if page.extract_text().strip():
                        return False  # Found text, so it's NOT scanned
                return True  # No text found on any page, assume scanned
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return True  # If we can't read it, treat it as scanned to be safe

    def process_pdf(pdf_path, output_dir="pdf_output"):
        """
        Main function: 
        - If scanned: converts PDF pages to JPG images.
        - If not scanned: extracts text to a .txt file.
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get base filename without extension
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        
        print(f"Processing: {pdf_path}")
        textOutput = ""
        if PDFProcessor.is_pdf_scanned(pdf_path):
            print("  → Scanned PDF detected. Converting to JPG images...")
            
            # Convert PDF pages to PIL images
            images = convert_from_path(pdf_path, dpi=500)  # 500 DPI is a good balance of quality/size
            count = 0
            
            for i, image in enumerate(images, start=1):
                # Save each page as a JPG
                image_path = os.path.join(output_dir, f"{base_name}_page_{i}.jpg")
                image.save(image_path, "JPEG", quality=95)
                res = EasyOCRWithRiskScore().extract_text_with_confidence(image_path, output=False, output_dir=output_dir)
                textOutput += f"<<<Page {i}>>>\n"
                textOutput += res['text'].strip() + "\n"
                count += res['risk_score']
                os.remove(image_path)


            txt_path = os.path.join(output_dir, f"{base_name}___{round(count/len(images), 0)}___.txt")
            with open(txt_path, 'w', encoding='utf-8') as txt_file:
                txt_file.write(textOutput.strip())
            
        else:
            print("  → Text-based PDF detected. Extracting text...")
            
            # Extract text using PyPDF2
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                full_text = ""
                
                for page_num, page in enumerate(reader.pages, start=1):
                    page_text = page.extract_text()
                    if page_text.strip():
                        full_text += f"<<<Page {page_num}>>>\n"
                        full_text += page_text + "\n\n"
            
            # Save text to a .txt file
            txt_path = os.path.join(output_dir, f"{base_name}___100___.txt")
            with open(txt_path, 'w', encoding='utf-8') as txt_file:
                txt_file.write(full_text)
            
        print(f"  ✅ Text extracted and saved to: {txt_path}")


# ====================
# USE THE SCRIPT
# ====================
if __name__ == "__main__":
    # Example: Replace with the path to your PDF
    pdf_file = "/home/cyy/Documents/SMUHack/CUAD_v1/full_contract_pdf/Part_II/Commercial Contracts (Part II-A)/Hosting/BANGIINC_05_25_2005-EX-10-Premium Managed Hosting Agreement.PDF"
    # Run the processing
    PDFProcessor.process_pdf(pdf_file, output_dir="pdf_output")