import os
import json
import boto3
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from typing import Dict, List, Union, Optional
import fitz  # PyMuPDF
import uuid
import warnings
import tempfile
from langchain_core.documents import Document
import chromadb
from chromadb.config import Settings
from models import InvoiceDetails, get_db, InvoiceItem, TaxDetails
from datetime import datetime
import re
from sqlalchemy.orm import Session
import pdf2image
from PIL import Image
import base64
from io import BytesIO
from fastapi import HTTPException

# Suppress warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Configuration
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')
AWS_REGION = os.getenv('AWS_REGION')
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# Initialize Gemini
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)

class DocumentProcessor:
    def __init__(self):
        self.gemini_model = genai.GenerativeModel('gemini-2.0-flash')
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        self.persist_directory = "chroma_db"
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Configure Chroma settings
        self.chroma_settings = Settings(
            anonymized_telemetry=False,
            allow_reset=True,
            is_persistent=True
        )

    def _encode_image_to_base64(self, image):
        """Convert PIL Image to base64 string"""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str

    def extract_invoice_details(self, image_or_text, is_image=False):
        """Extract key details from invoice using Gemini"""
        prompt = """
        Extract the following information from this invoice. If a field is not found, return "N/A" for text and 0 for numbers.
        Return the response in this exact JSON format:
        {
            "invoice_number": "value",
            "invoice_date": "YYYY-MM-DD",
            "due_date": "YYYY-MM-DD",
            "biller_company": "value",
            "biller_address": "complete address",
            "recipient_name": "name of person or company being billed",
            "recipient_address": "complete billing address",
            "items": [
                {
                    "item_name": "name",
                    "quantity": number,
                    "unit_price": number,
                    "total_price": number,
                    "description": "description if any"
                }
            ],
            "tax_details": [
                {
                    "tax_type": "GST/VAT/etc",
                    "tax_rate": number,
                    "tax_amount": number,
                    "description": "description if any"
                }
            ],
            "subtotal": number,
            "total_tax": number,
            "total_amount": number,
            "currency": "USD/EUR/etc",
            "payment_terms": "value"
        }
        """
        
        try:
            if is_image:
                response = self.gemini_model.generate_content([prompt, image_or_text])
            else:
                response = self.gemini_model.generate_content(f"{prompt}\n\nInvoice text:\n{image_or_text}")

            # Clean the response text to ensure valid JSON
            cleaned_text = response.text.strip()
            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text[7:-3]  # Remove ```json and ``` markers
            details = json.loads(cleaned_text)
            
            # Validate and clean the data
            items = details.get('items', [])
            if not items:
                items = [{
                    "item_name": "Unknown Item",
                    "quantity": details.get('total_amount', 0),
                    "unit_price": details.get('total_amount', 0),
                    "total_price": details.get('total_amount', 0),
                    "description": "N/A"
                }]

            tax_details = details.get('tax_details', [])
            if not tax_details:
                tax_details = [{
                    "tax_type": "N/A",
                    "tax_rate": 0,
                    "tax_amount": 0,
                    "description": "N/A"
                }]

            return {
                "invoice_number": str(details.get('invoice_number', 'N/A')),
                "invoice_date": details.get('invoice_date', '2000-01-01'),
                "due_date": details.get('due_date', '2000-01-01'),
                "biller_company": str(details.get('biller_company', 'N/A')),
                "biller_address": str(details.get('biller_address', 'N/A')),
                "recipient_name": str(details.get('recipient_name', 'N/A')),
                "recipient_address": str(details.get('recipient_address', 'N/A')),
                "items": items,
                "tax_details": tax_details,
                "subtotal": float(details.get('subtotal', 0)),
                "total_tax": float(details.get('total_tax', 0)),
                "total_amount": float(details.get('total_amount', 0)),
                "currency": str(details.get('currency', 'USD')),
                "payment_terms": str(details.get('payment_terms', 'N/A'))
            }
        except Exception as e:
            print(f"Error extracting invoice details: {str(e)}")
            return {
                "invoice_number": "N/A",
                "invoice_date": "2000-01-01",
                "due_date": "2000-01-01",
                "biller_company": "N/A",
                "biller_address": "N/A",
                "recipient_name": "N/A",
                "recipient_address": "N/A",
                "items": [],
                "tax_details": [],
                "subtotal": 0,
                "total_tax": 0,
                "total_amount": 0,
                "currency": "USD",
                "payment_terms": "N/A"
            }

    def save_invoice_details(self, document_id: str, details: dict, db: Session):
        """Save invoice details to database"""
        try:
            # Parse dates with fallback to default date if 'N/A'
            def parse_date(date_str):
                if date_str == 'N/A' or not date_str:
                    return datetime(2000, 1, 1)  # Default date
                try:
                    return datetime.strptime(date_str, '%Y-%m-%d')
                except ValueError:
                    return datetime(2000, 1, 1)  # Default date on parse error

            invoice_details = InvoiceDetails(
                document_id=document_id,
                generated_name=details.get('generated_name'),
                invoice_number=details.get('invoice_number', 'N/A'),
                invoice_date=parse_date(details.get('invoice_date')),
                due_date=parse_date(details.get('due_date')),
                biller_company=details.get('biller_company', 'N/A'),
                biller_address=details.get('biller_address', 'N/A'),
                recipient_name=details.get('recipient_name', 'N/A'),
                recipient_address=details.get('recipient_address', 'N/A'),
                subtotal=details.get('subtotal', 0),
                total_tax=details.get('total_tax', 0),
                total_amount=details.get('total_amount', 0),
                currency=details.get('currency', 'USD'),
                payment_terms=details.get('payment_terms', 'N/A')
            )
            
            # Add items
            for item in details.get('items', []):
                invoice_item = InvoiceItem(
                    document_id=document_id,
                    item_name=item.get('item_name', 'N/A'),
                    quantity=float(item.get('quantity', 0)),
                    unit_price=float(item.get('unit_price', 0)),
                    total_price=float(item.get('total_price', 0)),
                    description=item.get('description', 'N/A')
                )
                invoice_details.items.append(invoice_item)

            # Add tax details
            for tax in details.get('tax_details', []):
                tax_detail = TaxDetails(
                    document_id=document_id,
                    tax_type=tax.get('tax_type', 'N/A'),
                    tax_rate=float(tax.get('tax_rate', 0)),
                    tax_amount=float(tax.get('tax_amount', 0)),
                    description=tax.get('description', 'N/A')
                )
                invoice_details.tax_details.append(tax_detail)

            db.add(invoice_details)
            db.commit()
            return invoice_details
            
        except Exception as e:
            db.rollback()
            print(f"Error saving invoice details: {str(e)}")
            raise

    def generate_document_name(self, image_or_text, is_image=False):
        """Generate a descriptive name for the document using Gemini"""
        prompt = """
        Generate a concise but descriptive filename for this invoice document.
        The filename should follow this format: "<BillerCompany>_Invoice_<InvoiceNumber>_<Date>".
        If any part is not found, use a reasonable alternative.
        Keep it under 50 characters and use only alphanumeric characters, underscores, and hyphens.
        Do not include file extension.
        """
        
        try:
            if is_image:
                response = self.gemini_model.generate_content([prompt, image_or_text])
            else:
                response = self.gemini_model.generate_content(f"{prompt}\n\nInvoice text:\n{image_or_text}")

            name = response.text.strip()
            # Clean the filename
            name = re.sub(r'[^\w\-_]', '_', name)  # Replace invalid chars with underscore
            name = re.sub(r'_+', '_', name)  # Replace multiple underscores with single
            name = name[:50]  # Limit length
            return name
        except Exception as e:
            print(f"Error generating document name: {str(e)}")
            return f"Invoice_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def _extract_text_from_file(self, file_path: str) -> str:
        """Extract text from PDF or image file using Gemini vision"""
        file_ext = os.path.splitext(file_path.lower())[1]
        
        try:
            if file_ext == '.pdf':
                # Convert PDF to images
                pages = pdf2image.convert_from_path(file_path)
                if not pages:
                    raise ValueError("No pages found in PDF")
                
                full_text = ""
                
                # Process each page with Gemini
                for page in pages:
                    # Use Gemini to extract text
                    prompt = """
                    Extract all text from this image, preserving the exact formatting and numbers.
                    Include ALL text visible in the image, even if it seems unimportant.
                    Do not summarize or skip any text.
                    """
                    response = self.gemini_model.generate_content([prompt, page])
                    
                    if not response.text:
                        raise ValueError(f"Gemini returned empty text for page")
                    
                    full_text += response.text + "\n\n"
                
                if not full_text.strip():
                    raise ValueError("No text extracted from PDF")
                
                return full_text
            else:
                # Process single image
                image = Image.open(file_path)
                # Ensure the image is valid
                image.verify()
                # Reopen after verify
                image = Image.open(file_path)
                
                # Use Gemini to extract text
                prompt = """
                Extract all text from this image, preserving the exact formatting and numbers.
                Include ALL text visible in the image, even if it seems unimportant.
                Do not summarize or skip any text.
                """
                response = self.gemini_model.generate_content([prompt, image])
                
                if not response.text:
                    raise ValueError("Gemini returned empty text for image")
                
                return response.text
                
        except Exception as e:
            print(f"Error extracting text from file: {str(e)}")
            raise ValueError(f"Failed to extract text: {str(e)}")

    def process_document(self, file_path: str, file_name: str, db: Session) -> Dict:
        """Process a document and store its chunks in Chroma"""
        try:
            # Load the image/PDF for Gemini vision processing
            file_ext = os.path.splitext(file_path.lower())[1]
            is_image = file_ext != '.pdf'
            
            if is_image:
                image = Image.open(file_path)
                # Verify image is valid
                image.verify()
                # Reopen after verify
                image = Image.open(file_path)
            
            # Generate document ID
            document_id = str(uuid.uuid4())
            
            # Extract text using Gemini vision
            text = self._extract_text_from_file(file_path)
            
            if not text.strip():
                raise ValueError("No text could be extracted from the document")
            
            # Generate document name using Gemini vision
            if is_image:
                generated_name = self.generate_document_name(image, is_image=True)
            else:
                generated_name = self.generate_document_name(text)
            
            # Extract invoice details using Gemini vision
            if is_image:
                invoice_details = self.extract_invoice_details(image, is_image=True)
            else:
                invoice_details = self.extract_invoice_details(text)
            
            if invoice_details:
                invoice_details['generated_name'] = generated_name
                self.save_invoice_details(document_id, invoice_details, db)
            
            # Split text into chunks
            texts = self.text_splitter.split_text(text)
            
            if not texts:
                raise ValueError("Text splitting resulted in no chunks")
            
            # Create documents with metadata
            documents = [
                Document(
                    page_content=chunk,
                    metadata={
                        "document_id": document_id,
                        "source": file_name,
                        "chunk_id": f"chunk_{i}",
                        "page_number": i // 2 + 1
                    }
                ) for i, chunk in enumerate(texts)
            ]
            
            # Create Chroma collection for this document
            vectorstore = Chroma(
                collection_name=document_id,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
                client_settings=self.chroma_settings
            )
            
            # Add documents to the collection
            vectorstore.add_documents(documents)
            vectorstore.persist()
            
            # Upload to S3
            s3_key = f"documents/{document_id}/{file_name}"
            with open(file_path, 'rb') as file:
                s3_client.upload_fileobj(file, S3_BUCKET_NAME, s3_key)
            
            return {
                "document_id": document_id,
                "s3_key": s3_key,
                "name": file_name,
                "num_chunks": len(texts)
            }
            
        except ValueError as e:
            print(f"Validation error in process_document: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            print(f"Error in process_document: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

    def search_document(self, document_id: str, query: str) -> Dict:
        """Search within a specific document"""
        try:
            # Load the document's vector store
            vectorstore = Chroma(
                collection_name=document_id,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
                client_settings=self.chroma_settings
            )
            
            # Search for relevant chunks
            results = vectorstore.similarity_search_with_relevance_scores(
                query,
                k=5  # Number of results to return
            )
            
            # Format results
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    "content": doc.page_content,
                    "relevance": float(score),
                    "metadata": doc.metadata,
                    "page": doc.metadata.get("page_number", 1)
                })
            
            # Sort results by relevance score
            formatted_results.sort(key=lambda x: x["relevance"], reverse=True)
            
            # Generate AI explanation using Gemini
            context = "\n".join([
                f"[Page {r['page']}]: {r['content']}" 
                for r in formatted_results[:3]
            ])
            explanation = self._generate_explanation(query, context)
            
            return {
                "query": query,
                "results": formatted_results,
                "explanation": explanation,
                "total_results": len(formatted_results)
            }
            
        except Exception as e:
            raise Exception(f"Error searching document: {str(e)}")

    def _generate_explanation(self, query: str, context: str) -> str:
        """Generate an AI explanation for the search results"""
        prompt = f"""
        Based on the following document excerpts, provide a clear and concise answer to the query.
        If the context doesn't contain relevant information, say so.
        
        Query: {query}
        
        Document Excerpts:
        {context}
        
        Please provide a detailed answer, citing specific information from the document where possible.
        If you're not certain about something, say so.
        
        Answer:
        """
        
        try:
            response = self.gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating explanation: {str(e)}"

    def delete_document(self, document_id: str) -> None:
        """Delete a document's vectors and S3 file"""
        try:
            # Delete from Chroma
            vectorstore = Chroma(
                collection_name=document_id,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
                client_settings=self.chroma_settings
            )
            vectorstore.delete_collection()
            vectorstore.persist()
            
            # Delete from S3
            response = s3_client.list_objects_v2(
                Bucket=S3_BUCKET_NAME,
                Prefix=f"documents/{document_id}/"
            )
            
            if 'Contents' in response:
                for obj in response['Contents']:
                    s3_client.delete_object(
                        Bucket=S3_BUCKET_NAME,
                        Key=obj['Key']
                    )
                    
        except Exception as e:
            raise Exception(f"Error deleting document: {str(e)}")

class DocumentExtractorAPI:
    def __init__(self):
        self.processor = DocumentProcessor()
        
    def process_document(self, file_path: str, db: Session) -> Dict:
        """Process a new document"""
        file_name = os.path.basename(file_path)
        return self.processor.process_document(file_path, file_name, db)
        
    def search_document(self, document_id: str, query: str) -> Dict:
        """Search within a specific document"""
        return self.processor.search_document(document_id, query)
        
    def delete_document(self, document_id: str) -> None:
        """Delete a document"""
        return self.processor.delete_document(document_id)

# Example usage
if __name__ == "__main__":
    api = DocumentExtractorAPI()
    
    # Example: Process a local PDF file
    pdf_path = "/Users/pi-in-140/Downloads/Invoice-989B3CF6-0013.pdf"
    result = api.process_document(pdf_path)
    print(json.dumps(result, indent=2))
    
    # Example: Perform a search
    search_results = api.search_document(result["document_id"], "Find all items with quantity greater than 10")
    print(json.dumps(search_results, indent=2))
    
    # Example: Delete document
    api.delete_document(result["document_id"])