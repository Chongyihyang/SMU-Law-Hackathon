import easyocr
import numpy as np
import os

class EasyOCRWithRiskScore:
    def __init__(self):
        # Initialize EasyOCR for English
        self.reader = easyocr.Reader(['en'], gpu=False)
    
    def extract_text_with_confidence(self, image_path, output=True, output_dir="pdf_output"):
        """
        Extract text with confidence scores from EasyOCR
        """
        # Read image
        result = self.reader.readtext(image_path, 
                                      paragraph=False,
                                      detail=1,  # Get confidence details
                                      min_size=10)
        
        # Extract text and confidence
        full_text = ""
        confidences = []
        word_details = []
        
        for detection in result:
            bbox, text, confidence = detection
            full_text += text + " "
            confidences.append(confidence * 100)
            word_details.append({
                'text': text,
                'confidence': confidence * 100,
                'bbox': bbox
            })
        
        # Calculate risk score
        if confidences:
            avg_confidence = np.mean(confidences)
            risk_score = 100 - avg_confidence
        else:
            risk_score = 100

        if output:
            # Save text to a .txt file
            txt_path = os.path.join(output_dir, f"{image_path}__{round(risk_score, 0)}__.txt")
            with open(txt_path, 'w', encoding='utf-8') as txt_file:
                txt_file.write(full_text.strip())
            
            print(f"  ✅ Text extracted and saved to: {txt_path}")

        return {
            'text': full_text.strip(),
            'risk_score': round(risk_score, 2),
            'avg_confidence': round(avg_confidence, 2) if confidences else 0,
            'word_details': word_details
        }