from fastapi import FastAPI, UploadFile, Request, Depends, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import Optional, Dict, List
import google.generativeai as genai
from neo4j import GraphDatabase
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
app = FastAPI(title="Neo4j Gemini Agent API")

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

# Neo4j Configuration
NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
NEO4J_USER = os.getenv('NEO4J_USER', 'neo4j')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', 'postgres')

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


class AgentResponse(BaseModel):
    query: str
    cypher_query: str
    results: List[Dict]
    total_results: int
    explanation: Optional[str] = None
    query_confidence: Optional[float] = None
    token_counts: Optional[Dict[str, int]] = None


class TokenCounter:
    def __init__(self):
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        
    def update(self, prompt_tokens: int, completion_tokens: int):
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens = self.prompt_tokens + self.completion_tokens
        
    def get_counts(self) -> Dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens
        }

# Initialize token counter
token_counter = TokenCounter()

# Database dependency
def get_neo4j_driver():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        yield driver
    finally:
        driver.close()


# Schema retrieval function that doesn't rely on APOC
def get_schema(driver):
    with driver.session() as session:
        # Get node labels
        node_labels_query = """
        MATCH (n) 
        WITH DISTINCT labels(n) AS labels
        UNWIND labels AS label
        RETURN DISTINCT label
        """
        node_labels = [record["label"] for record in session.run(node_labels_query)]

        # Get node properties for each label
        node_schema = {}
        for label in node_labels:
            props_query = f"""
            MATCH (n:{label})
            UNWIND keys(n) AS key
            RETURN DISTINCT key
            LIMIT 100
            """
            properties = [record["key"] for record in session.run(props_query)]
            node_schema[label] = {"properties": {prop: "unknown" for prop in properties}}

        # Get relationship types
        rel_query = """
        MATCH ()-[r]->()
        RETURN DISTINCT type(r) AS relType
        """
        relationship_types = [record["relType"] for record in session.run(rel_query)]

        # Get relationship source and target for each type
        rel_schema = []
        for rel_type in relationship_types:
            source_target_query = f"""
            MATCH (s)-[r:{rel_type}]->(t)
            RETURN DISTINCT labels(s)[0] AS sourceLabel, labels(t)[0] AS targetLabel
            LIMIT 5
            """
            for record in session.run(source_target_query):
                rel_schema.append({
                    "relType": rel_type,
                    "sourceLabel": record["sourceLabel"],
                    "targetLabel": record["targetLabel"]
                })

        return {"nodes": node_schema, "relationships": rel_schema}


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


@app.delete("/document/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and its vectors"""
    try:
        # Get document info first to get s3_key
        client = chromadb.PersistentClient(
            path=doc_api.processor.persist_directory,
            settings=doc_api.processor.chroma_settings
        )
        
        try:
            # Get S3 object info before deleting
            response = s3_client.list_objects_v2(
                Bucket=S3_BUCKET_NAME,
                Prefix=f"documents/{document_id}/"
            )
            
            # Delete document using the API
            doc_api.delete_document(document_id)
            
            return {"message": "Document deleted successfully"}
            
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {str(e)}")
            raise HTTPException(status_code=404, detail="Document not found")

    except Exception as e:
        logger.error(f"Error in delete operation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/schema")
async def get_database_schema(driver: GraphDatabase.driver = Depends(get_neo4j_driver)):
    """
    Get the database schema information
    """
    try:
        schema = get_schema(driver)
        return JSONResponse(content=schema)
    except Exception as e:
        logger.error(f"Error getting schema: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving schema: {str(e)}"
        )


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
            return {
                "invoice_number": "N/A",
                "invoice_date": "N/A",
                "due_date": "N/A",
                "biller_company": "N/A",
                "biller_address": "N/A",
                "billed_to_company": "N/A",
                "billed_to_address": "N/A",
                "total_amount": 0,
                "currency": "N/A",
                "payment_terms": "N/A",
                "items": [],
                "tax_details": [],
                "subtotal": 0,
                "total_tax": 0,
                "total_amount": 0
            }
            
        return {
            "invoice_number": invoice_details.invoice_number,
            "invoice_date": invoice_details.invoice_date.strftime('%Y-%m-%d'),
            "due_date": invoice_details.due_date.strftime('%Y-%m-%d'),  # Fixed date format
            "biller_company": invoice_details.biller_company,
            "biller_address": invoice_details.biller_address,
            "billed_to_company": invoice_details.billed_to_company,
            "billed_to_address": invoice_details.billed_to_address,
            "total_amount": float(invoice_details.total_amount),  # Ensure float conversion
            "currency": invoice_details.currency,
            "payment_terms": invoice_details.payment_terms,
            "items": [
                {
                    "item_name": item.item_name,
                    "quantity": item.quantity,
                    "unit_price": item.unit_price,
                    "total_price": item.total_price,
                    "description": item.description
                } for item in invoice_details.items
            ],
            "tax_details": [
                {
                    "tax_type": tax.tax_type,
                    "tax_rate": tax.tax_rate,
                    "tax_amount": tax.tax_amount,
                    "description": tax.description
                } for tax in invoice_details.tax_details
            ],
            "subtotal": float(invoice_details.subtotal),
            "total_tax": float(invoice_details.total_tax),
            "total_amount": float(invoice_details.total_amount)
        }
    except Exception as e:
        logger.error(f"Error fetching invoice details: {str(e)}")
        return {
            "invoice_number": "Error",
            "invoice_date": "N/A",
            "due_date": "N/A",
            "biller_company": "N/A",
            "biller_address": "N/A",
            "billed_to_company": "N/A",
            "billed_to_address": "N/A",
            "total_amount": 0,
            "currency": "N/A",
            "payment_terms": "N/A",
            "items": [],
            "tax_details": [],
            "subtotal": 0,
            "total_tax": 0,
            "total_amount": 0
        }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)