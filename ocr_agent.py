from dataclasses import dataclass
from typing import List
from PIL import Image
from pytesseract import pytesseract
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from pydantic_ai.models.ollama import OllamaModel
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from datetime import date

from dataclasses import dataclass

import nest_asyncio
nest_asyncio.apply()

# Set the Tesseract OCR path (adjust as necessary for your system)
pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'  

model = OllamaModel(
    model_name='qwen2.5:7b',  
    base_url='http://localhost:11434/v1/',  
)

@dataclass
class OCRDependencies:
    """Dependencies required for OCR processing."""
    pdf_path: str  # Path to the PDF to be processed


class OCRResult(BaseModel):
    """Structured representation of extracted text."""
    page_number: int = Field(..., description="Page number in the PDF.")
    extracted_text: str = Field(..., description="Text extracted from the page.")


ocr_agent = Agent(
    model,
    deps_type=OCRDependencies,
    result_type=List[OCRResult],
    system_prompt="You are an OCR agent that reads PDFs and extracts all text in a structured format.",
)


@ocr_agent.tool
def extract_text_from_pdf(ctx: RunContext[OCRDependencies]) -> List[OCRResult]:
    """
    Extract text from a PDF file using OCR.
    
    Args:
        ctx: The context with dependencies.
    
    Returns:
        List[OCRResult]: List of extracted text per page.
    """
    pdf_path = ctx.deps.pdf_path
    results = []

    # Convert PDF pages to images
    images = convert_from_path(pdf_path)
    for page_number, image in enumerate(images, start=1):
        text = pytesseract.image_to_string(image)
        results.append(OCRResult(page_number=page_number, extracted_text=text))

    return results



# Example usage
if __name__ == "__main__":
    deps = OCRDependencies(pdf_path="test.pdf")
    results = ocr_agent.run_sync("Extract text from the provided PDF.", deps=deps)
    for result in results:
        print(f"Page {result.page_number}:")
        print(result.extracted_text)
        print("-" * 50)
