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
import pytesseract
from PIL import Image
import pdf2image
import io

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
        self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
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

    def extract_invoice_details(self, text: str, file_path: str = None) -> dict:
        """Extract key details from invoice using Gemini"""
        try:
            # For both PDFs and images, use direct image processing
            if file_path:
                # For PDFs, process first page
                if file_path.lower().endswith('.pdf'):
                    pages = pdf2image.convert_from_path(file_path)
                    if pages:
                        # Convert first page to bytes
                        img_byte_arr = io.BytesIO()
                        pages[0].save(img_byte_arr, format='PNG')
                        image_data = img_byte_arr.getvalue()
                else:
                    # For images, read directly
                    with open(file_path, 'rb') as img_file:
                        image_data = img_file.read()

                prompt = """
                You are a precise invoice data extractor. Analyze this invoice image carefully and extract the following information.
                Look for these specific elements:

                1. Invoice Number: Usually labeled as "Invoice #", "Invoice Number", or similar
                2. Dates: Look for "Invoice Date", "Due Date", "Issue Date" - ensure YYYY-MM-DD format
                3. Companies: 
                    - Biller: The company issuing the invoice (look for logo, header, or "From" section)
                    - Recipient: The company being billed (look for "Bill To", "To", or similar)
                4. Addresses: Complete addresses for both companies
                5. Line Items: Look for itemized list of products/services
                6. Tax Information: Look for VAT, GST, or other tax-related entries
                7. Payment Terms: Look for "Terms", "Payment Terms", or similar
                8. Amounts: Pay special attention to:
                    - Individual item prices and quantities
                    - Subtotal
                    - Tax amounts
                    - Total amount
                9. Currency: Look for currency symbols or codes

                Return the data in this exact JSON format:
                {
                    "invoice_number": "exact number as shown",
                    "invoice_date": "YYYY-MM-DD",
                    "due_date": "YYYY-MM-DD",
                    "biller_company": "complete company name",
                    "biller_address": "complete address",
                    "billed_to_company": "complete company name",
                    "billed_to_address": "complete address",
                    "items": [
                        {
                            "item_name": "exact item name",
                            "quantity": number,
                            "unit_price": number,
                            "total_price": number,
                            "description": "any additional details"
                        }
                    ],
                    "tax_details": [
                        {
                            "tax_type": "exact tax type (GST/VAT/etc)",
                            "tax_rate": number,
                            "tax_amount": number,
                            "description": "any tax-related notes"
                        }
                    ],
                    "subtotal": number,
                    "total_tax": number,
                    "total_amount": number,
                    "currency": "USD/EUR/etc",
                    "payment_terms": "exact payment terms"
                }

                Important:
                - Extract EXACT values as shown in the invoice
                - Maintain precise number formatting
                - Use "N/A" for missing text fields
                - Use 0 for missing numerical values
                - Ensure dates are in YYYY-MM-DD format
                - Include ALL found line items
                - Include ALL tax details
                """

                # Process with Gemini
                response = self.gemini_model.generate_content([
                    prompt,
                    {"mime_type": "image/png", "data": image_data}
                ])
            else:
                # Fallback to text-based processing if no file path
                prompt = f"""
                Extract the following information from this invoice text.
                Be extremely precise and thorough.

                Text content:
                {text}
                """
            response = self.gemini_model.generate_content(prompt)

            # Clean and parse the response
            cleaned_text = response.text.strip()
            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text[7:-3]
            
            # Try to fix common JSON formatting issues
            cleaned_text = cleaned_text.replace("'", '"')
            cleaned_text = re.sub(r'(\d+)\.?0+([,\s}])', r'\1\2', cleaned_text)
            
            try:
                details = json.loads(cleaned_text)
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {str(e)}")
                print(f"Problematic text: {cleaned_text}")
                raise

            # Validate and clean the data
            items = details.get('items', [])
            if not items:
                items = [{
                    "item_name": "Unknown Item",
                    "quantity": 1,
                    "unit_price": float(details.get('total_amount', 0)),
                    "total_price": float(details.get('total_amount', 0)),
                    "description": "N/A"
                }]

            # Ensure all numerical values are properly converted
            for item in items:
                item['quantity'] = float(item.get('quantity', 0))
                item['unit_price'] = float(item.get('unit_price', 0))
                item['total_price'] = float(item.get('total_price', 0))

            tax_details = details.get('tax_details', [])
            if not tax_details:
                tax_details = [{
                    "tax_type": "N/A",
                    "tax_rate": 0,
                    "tax_amount": 0,
                    "description": "N/A"
                }]

            # Ensure all tax values are properly converted
            for tax in tax_details:
                tax['tax_rate'] = float(tax.get('tax_rate', 0))
                tax['tax_amount'] = float(tax.get('tax_amount', 0))

            # Format dates properly
            invoice_date = details.get('invoice_date', '2000-01-01')
            due_date = details.get('due_date', '2000-01-01')
            
            # Validate dates
            try:
                datetime.strptime(invoice_date, '%Y-%m-%d')
            except ValueError:
                invoice_date = '2000-01-01'
            
            try:
                datetime.strptime(due_date, '%Y-%m-%d')
            except ValueError:
                due_date = '2000-01-01'

            return {
                "invoice_number": str(details.get('invoice_number', 'N/A')),
                "invoice_date": invoice_date,
                "due_date": due_date,
                "biller_company": str(details.get('biller_company', 'N/A')),
                "biller_address": str(details.get('biller_address', 'N/A')),
                "billed_to_company": str(details.get('billed_to_company', 'N/A')),
                "billed_to_address": str(details.get('billed_to_address', 'N/A')),
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
                "billed_to_company": "N/A",
                "billed_to_address": "N/A",
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
            invoice_details = InvoiceDetails(
                document_id=document_id,
                generated_name=details.get('generated_name'),
                invoice_number=details.get('invoice_number', 'N/A'),
                invoice_date=datetime.strptime(details.get('invoice_date', '2000-01-01'), '%Y-%m-%d'),
                due_date=datetime.strptime(details.get('due_date', '2000-01-01'), '%Y-%m-%d'),
                biller_company=details.get('biller_company', 'N/A'),
                biller_address=details.get('biller_address', 'N/A'),
                billed_to_company=details.get('billed_to_company', 'N/A'),
                billed_to_address=details.get('billed_to_address', 'N/A'),
                subtotal=details.get('subtotal', 0),
                total_tax=details.get('total_tax', 0),
                total_amount=details.get('total_amount', 0)
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
            
        except Exception as e:
            db.rollback()
            print(f"Error saving invoice details: {str(e)}")

    def generate_document_name(self, text: str) -> str:
        """Generate a descriptive name for the document using Gemini"""
        prompt = f"""
        Generate a concise but descriptive filename for this invoice document.
        The filename should follow this format: "<BillerCompany>_Invoice_<InvoiceNumber>_<Date>".
        If any part is not found, use a reasonable alternative.
        Keep it under 50 characters and use only alphanumeric characters, underscores, and hyphens.
        Do not include file extension.

        Invoice text:
        {text}
        """
        
        try:
            response = self.gemini_model.generate_content(prompt)
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
        """Extract text from PDF or image file using only Gemini"""
        file_ext = os.path.splitext(file_path.lower())[1]
        
        try:
            if file_ext == '.pdf':
                # For PDFs, first convert to images
                pages = pdf2image.convert_from_path(file_path)
                all_text = []
                
                # Process each page as an image using Gemini
                for page in pages:
                    # Convert page to bytes
                    img_byte_arr = io.BytesIO()
                    page.save(img_byte_arr, format='PNG')
                    img_byte_arr = img_byte_arr.getvalue()
                    
                    # Process with Gemini
                    prompt = """
                    Extract all text content from this invoice image.
                    Include all text, numbers, dates, and details you can see.
                    Maintain the original structure and formatting as much as possible.
                    """
                    
                    response = self.gemini_model.generate_content([
                        prompt,
                        {"mime_type": "image/png", "data": img_byte_arr}
                    ])
                    
                    if response.text:
                        all_text.append(response.text)
                
                return "\n\n".join(all_text)
            else:
                # For images, use Gemini directly
                with open(file_path, 'rb') as img_file:
                    image_data = img_file.read()
                
                prompt = """
                Extract all text content from this invoice image.
                Include all text, numbers, dates, and details you can see.
                Maintain the original structure and formatting as much as possible.
                """
                
                response = self.gemini_model.generate_content([
                    prompt,
                    {"mime_type": "image/jpeg", "data": image_data}
                ])
                
                return response.text if response.text else "No text could be extracted from the image"
                
        except Exception as e:
            print(f"Error in text extraction: {str(e)}")
            return "Error extracting text from document"

    def process_document(self, file_path: str, file_name: str, db: Session) -> Dict:
        """Process a document and store its chunks in Chroma"""
        try:
            # Extract text from file (PDF or image)
            text = self._extract_text_from_file(file_path)
            
            if not text or text.isspace():
                raise Exception("No text could be extracted from the document")
            
            # Split text into chunks
            texts = self.text_splitter.split_text(text)
            
            if not texts:
                texts = [text]  # Use the entire text as one chunk if splitting fails
            
            # Generate document ID
            document_id = str(uuid.uuid4())
            
            # Generate document name
            generated_name = self.generate_document_name(text)
            
            # Extract and save invoice details with generated name
            invoice_details = self.extract_invoice_details(text, file_path)
            if invoice_details:
                invoice_details['generated_name'] = generated_name
                self.save_invoice_details(document_id, invoice_details, db)
            
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
                if chunk and not chunk.isspace()  # Only include non-empty chunks
            ]
            
            if not documents:
                raise Exception("No valid text chunks could be created")
            
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
            
        except Exception as e:
            print(f"Error in process_document: {str(e)}")
            raise Exception(f"Error processing document: {str(e)}")

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