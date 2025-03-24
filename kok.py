from fastapi import FastAPI, UploadFile, Request, Depends, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import Optional, Dict, List
import google.generativeai as genai
import os
from dotenv import load_dotenv
from pydantic import BaseModel
import json
import logging
from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import io
from urllib.parse import unquote
import boto3
import tempfile
import chromadb
from models import get_db, InvoiceDetails
from sqlalchemy.orm import Session

# Import the DocumentExtractorAPI from main.py
from main import DocumentExtractorAPI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Document Analysis API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize DocumentExtractorAPI
doc_api = DocumentExtractorAPI()

# Initialize Gemini
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# S3 Configuration
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')

# Add after S3_BUCKET_NAME declaration
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION')
)

# Test S3 connection at startup
try:
    s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME, MaxKeys=1)
    logger.info("Successfully connected to S3")
except Exception as e:
    logger.error(f"Failed to connect to S3: {str(e)}")

# Pydantic models
class SearchRequest(BaseModel):
    query: str
    params: Optional[Dict] = None
    context: Optional[str] = None
    document_id: str

class SearchResponse(BaseModel):
    query: str
    results: List[Dict]
    total_results: int
    explanation: Optional[str] = None

@app.post("/process-document")
async def process_document(file: UploadFile, db: Session = Depends(get_db)):
    """Process and store a new document"""
    try:
        # Check file extension
        allowed_extensions = {'.pdf', '.jpg', '.jpeg', '.png'}
        file_ext = os.path.splitext(file.filename.lower())[1]
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail="Only PDF and image files (JPG, JPEG, PNG) are supported"
            )

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            # Process document with db session
            result = doc_api.process_document(tmp_path, db)
            os.unlink(tmp_path)
            return result
        except Exception as e:
            os.unlink(tmp_path)
            raise e
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/document/{document_id}")
async def delete_document(document_id: str, db: Session = Depends(get_db)):
    """Delete a document and its associated data"""
    try:
        # Delete from database first
        invoice_details = db.query(InvoiceDetails).filter(
            InvoiceDetails.document_id == document_id
        ).first()
        
        if invoice_details:
            db.delete(invoice_details)
            db.commit()
        
        # Delete from Chroma
        try:
            client = chromadb.PersistentClient(
                path=doc_api.processor.persist_directory,
                settings=doc_api.processor.chroma_settings
            )
            collection = client.get_collection(name=document_id)
            if collection:
                client.delete_collection(name=document_id)
        except Exception as e:
            logger.error(f"Error deleting Chroma collection: {str(e)}")
            
        # Delete from S3
        try:
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
            logger.error(f"Error deleting S3 objects: {str(e)}")
            
        return {"message": "Document and associated data deleted successfully"}
            
    except Exception as e:
        logger.error(f"Error in delete operation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
async def get_documents(db: Session = Depends(get_db)):
    """Get list of all documents"""
    try:
        # Get all collections
        client = chromadb.PersistentClient(
            path=doc_api.processor.persist_directory,
            settings=doc_api.processor.chroma_settings
        )
        
        # In v0.6.0, list_collections() returns collection names as strings
        collection_names = client.list_collections()
        documents = []
        
        for collection_name in collection_names:
            # Get invoice details for generated name
            invoice_details = db.query(InvoiceDetails).filter(
                InvoiceDetails.document_id == collection_name
            ).first()
            
            # Get S3 object info
            response = s3_client.list_objects_v2(
                Bucket=S3_BUCKET_NAME,
                Prefix=f"documents/{collection_name}/"
            )
            
            if 'Contents' in response:
                s3_key = response['Contents'][0]['Key']
                original_name = s3_key.split('/')[-1]
                
                documents.append({
                    "id": collection_name,
                    "name": invoice_details.generated_name if invoice_details else original_name,
                    "original_name": original_name,
                    "s3_key": s3_key
                })
        
        # Sort documents by name
        documents.sort(key=lambda x: x["name"].lower())
        return documents
        
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search-graph")
async def search_graph(request: SearchRequest):
    """Search within a specific document"""
    try:
        result = doc_api.search_document(
            document_id=request.document_id,
            query=request.query
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/document/{s3_key:path}")
async def get_document(s3_key: str):
    """Get document file from S3"""
    try:
        # Get the file from S3
        response = s3_client.get_object(
            Bucket=S3_BUCKET_NAME,
            Key=s3_key
        )
        
        # Determine media type based on file extension
        file_ext = s3_key.lower().split('.')[-1]
        media_type = {
            'pdf': 'application/pdf',
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg',
            'png': 'image/png'
        }.get(file_ext, 'application/octet-stream')
        
        # Return the file as a streaming response
        return StreamingResponse(
            response['Body'].iter_chunks(),
            media_type=media_type,
            headers={
                'Content-Disposition': f'inline; filename="{s3_key.split("/")[-1]}"'
            }
        )
    except Exception as e:
        logger.error(f"Error getting document: {str(e)}")
        raise HTTPException(status_code=404, detail="Document not found")

@app.get("/document-info/{document_id}")
async def get_document_info(document_id: str):
    """Get document information"""
    try:
        # Get Chroma client
        client = chromadb.PersistentClient(
            path=doc_api.processor.persist_directory,
            settings=doc_api.processor.chroma_settings
        )
        
        try:
            # Get invoice details for the generated name
            db = next(get_db())
            invoice_details = db.query(InvoiceDetails).filter(
                InvoiceDetails.document_id == document_id
            ).first()
            
            # Just check if collection exists
            client.get_or_create_collection(name=document_id)
            
            # Get S3 object info
            response = s3_client.list_objects_v2(
                Bucket=S3_BUCKET_NAME,
                Prefix=f"documents/{document_id}/"
            )
            
            if 'Contents' in response:
                s3_key = response['Contents'][0]['Key']
                original_name = s3_key.split('/')[-1]
                return {
                    "id": document_id,
                    "name": invoice_details.generated_name if invoice_details else original_name,
                    "original_name": original_name,
                    "s3_key": s3_key
                }
            else:
                raise HTTPException(status_code=404, detail="Document not found in S3")
                
        except Exception as e:
            logger.error(f"Error getting collection {document_id}: {str(e)}")
            raise HTTPException(status_code=404, detail="Document not found")
            
    except Exception as e:
        logger.error(f"Error fetching document info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/invoice-details/{document_id}")
async def get_invoice_details(document_id: str, db: Session = Depends(get_db)):
    """Get invoice details for a document"""
    try:
        invoice_details = db.query(InvoiceDetails).filter(
            InvoiceDetails.document_id == document_id
        ).first()
        
        if not invoice_details:
            return {}
            
        # Helper function to check if a value is valid
        def is_valid(value):
            if value is None:
                return False
            if isinstance(value, str) and value in ['N/A', '']:
                return False
            if isinstance(value, (int, float)) and value == 0:
                return False
            return True

        # Build response with only valid fields
        response = {}
        
        # Add basic fields if they are valid
        fields = {
            'invoice_number': invoice_details.invoice_number,
            'biller_company': invoice_details.biller_company,
            'biller_address': invoice_details.biller_address,
            'recipient_name': invoice_details.recipient_name,
            'recipient_address': invoice_details.recipient_address,
            'payment_terms': invoice_details.payment_terms,
            'currency': invoice_details.currency
        }
        
        for key, value in fields.items():
            if is_valid(value):
                response[key] = value

        # Add date fields if they are valid (not default date)
        if invoice_details.invoice_date and invoice_details.invoice_date.year != 2000:
            response['invoice_date'] = invoice_details.invoice_date.strftime('%Y-%m-%d')
            
        if invoice_details.due_date and invoice_details.due_date.year != 2000:
            response['due_date'] = invoice_details.due_date.strftime('%Y-%m-%d')

        # Add amount fields if they are valid
        if is_valid(invoice_details.subtotal):
            response['subtotal'] = float(invoice_details.subtotal)
            
        if is_valid(invoice_details.total_tax):
            response['total_tax'] = float(invoice_details.total_tax)
            
        if is_valid(invoice_details.total_amount):
            response['total_amount'] = float(invoice_details.total_amount)

        # Add items if they exist and have valid data
        valid_items = [
            {
                key: value
                for key, value in {
                    "item_name": item.item_name,
                    "quantity": item.quantity,
                    "unit_price": item.unit_price,
                    "total_price": item.total_price,
                    "description": item.description
                }.items()
                if is_valid(value)
            }
            for item in invoice_details.items
        ]
        
        if valid_items:
            response['items'] = [item for item in valid_items if item]  # Only include items with data

        # Add tax details if they exist and have valid data
        valid_tax_details = [
            {
                key: value
                for key, value in {
                    "tax_type": tax.tax_type,
                    "tax_rate": tax.tax_rate,
                    "tax_amount": tax.tax_amount,
                    "description": tax.description
                }.items()
                if is_valid(value)
            }
            for tax in invoice_details.tax_details
        ]
        
        if valid_tax_details:
            response['tax_details'] = [tax for tax in valid_tax_details if tax]  # Only include tax details with data

        return response
        
    except Exception as e:
        logger.error(f"Error fetching invoice details: {str(e)}")
        return {}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)