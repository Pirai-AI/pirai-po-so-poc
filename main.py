import os
import json
import boto3
from dotenv import load_dotenv
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from typing import Dict, List, Union, Optional, Any
import fitz  # PyMuPDF
import re
import uuid
import warnings
import tempfile

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging

# Load environment variables
load_dotenv()

# AWS S3 Configuration
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')
AWS_REGION = os.getenv('AWS_REGION')
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')

# OpenAI API Key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Neo4j Configuration
NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
NEO4J_USER = os.getenv('NEO4J_USER', 'neo4j')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', 'postgres')

# Initialize sentence transformer model
model_name = "all-MiniLM-L6-v2"  # Can be customized based on requirements
embedding_model = SentenceTransformer(model_name)

# Initialize S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)

# Initialize Neo4j driver
neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

class DynamicDocumentExtractor:
    def __init__(self):
        pass
        
    def preprocess_document(self, input_data: Union[str, bytes], input_type: str) -> str:
        """
        Preprocess the document based on input type (text, pdf)
        Returns extracted text
        """
        if input_type == 'text':
            return input_data
        
        elif input_type == 'pdf':
            try:
                # Input data should be the local path to PDF file
                pdf_path = input_data
                text = ""
                with fitz.open(pdf_path) as doc:
                    for page in doc:
                        text += page.get_text()
                return text
            except Exception as e:
                return ""
        else:
            return ""

    def generate_embeddings(self, text: str) -> List[float]:
        """Generate embeddings for text using SentenceTransformer"""
        try:
            return embedding_model.encode(text).tolist()
        except Exception as e:
            return []

    def identify_document_type(self, text: str) -> Dict:
        """Use Gemini to identify document type and key structure"""
        prompt = f"""
        Analyze the following document text and determine its type and structure.
        Identify what kind of document this is (e.g., purchase order, sales order, invoice, contract, etc.)
        and what primary entities and data points are contained within it.
        
        For each entity or data point, identify:
        1. The entity/data point name
        2. The type of information it contains
        3. Its relationships to other entities if applicable
        
        Document Text:
        {text[:4000]}  # Limit text length for API
        
        Provide your analysis in this JSON format:
        {{
            "document_type": "identified document type",
            "entities": [
                {{
                    "name": "entity name",
                    "type": "entity type",
                    "attributes": ["attribute1", "attribute2"],
                    "relationships": [
                        {{"related_to": "other entity name", "relationship_type": "type of relationship"}}
                    ]
                }}
            ],
            "primary_identifiers": ["key1", "key2"]
        }}
        """
        
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            # Extract JSON from the response
            json_text = response.choices[0].message.content
            if '```json' in json_text:
                json_text = json_text.split('```json')[1].split('```')[0].strip()
            elif '```' in json_text:
                json_text = json_text.split('```')[1].split('```')[0].strip()
            return json.loads(json_text)
        except Exception as e:
            return {"document_type": "unknown", "entities": [], "primary_identifiers": []}

    def extract_structured_information(self, text: str, document_schema: Dict) -> Dict:
        """Extract structured information based on document schema"""
        # Create a detailed prompt based on the document schema
        entities_description = "\n".join([
            f"- {entity['name']}: {entity['type']} with attributes {', '.join(entity['attributes'])}"
            for entity in document_schema.get('entities', [])
        ])
        
        prompt = f"""
        Given the following document text and schema, extract all relevant information according to the schema structure.
        
        Document Type: {document_schema.get('document_type', 'unknown')}
        
        Primary Identifiers: {', '.join(document_schema.get('primary_identifiers', []))}
        
        Entities to Extract:
        {entities_description}
        
        Document Text:
        {text[:6000]}  # Limit text length for API
        
        Extract all information matching the schema above and provide a complete JSON output with all identified entities,
        their attributes, and values. Use null for missing values. Convert all monetary values to numbers, dates to ISO format,
        and ensure consistent formatting. For lists of items, extract all items found in the document.
        """
        
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            # Extract JSON from the response
            json_text = response.choices[0].message.content
            if '```json' in json_text:
                json_text = json_text.split('```json')[1].split('```')[0].strip()
            elif '```' in json_text:
                json_text = json_text.split('```')[1].split('```')[0].strip()
            
            extracted_data = json.loads(json_text)
            # Add document_type from schema
            extracted_data['document_type'] = document_schema.get('document_type', 'unknown')
            return extracted_data
        except Exception as e:
            return {"document_type": document_schema.get('document_type', 'unknown'), "error": str(e)}

    def normalize_value(self, value: Any) -> Any:
        """Normalize values for consistent storage in Neo4j"""
        if isinstance(value, str):
            # Try to convert string numbers to float
            if re.match(r'^\$?[\d,]+(\.\d+)?$', value.strip()):
                try:
                    return float(value.strip().replace('$', '').replace(',', ''))
                except:
                    pass
            # Try to normalize dates (basic implementation)
            if re.match(r'^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$', value.strip()):
                return value.strip()
        return value

    def create_dynamic_knowledge_graph(self, document_id: str, extracted_data: Dict, document_schema: Dict) -> None:
        """Create knowledge graph from dynamically extracted data using Neo4j"""
        with neo4j_driver.session() as session:
            # Create Document node
            session.run(
                """
                CREATE (d:Document {id: $document_id, type: $document_type})
                """,
                document_id=document_id,
                document_type=extracted_data.get('document_type', 'unknown')
            )
            
            # Process all entities based on dynamic schema
            for entity_schema in document_schema.get('entities', []):
                entity_name = entity_schema['name']
                entity_type = entity_schema['type']
                
                # Check if entity exists in extracted data
                if entity_name in extracted_data:
                    entity_data = extracted_data[entity_name]
                    
                    # Handle list of entities
                    if isinstance(entity_data, list):
                        for i, item in enumerate(entity_data):
                            self._create_entity_node(session, document_id, entity_type, item, f"{entity_name}_{i}")
                    else:
                        # Handle single entity
                        self._create_entity_node(session, document_id, entity_type, entity_data, entity_name)
            
            # Create relationships based on schema
            for entity_schema in document_schema.get('entities', []):
                for relationship in entity_schema.get('relationships', []):
                    source_entity = entity_schema['name']
                    target_entity = relationship['related_to']
                    relationship_type = relationship['relationship_type'].upper().replace(' ', '_')
                    
                    # Create relationship if both entities exist
                    if source_entity in extracted_data and target_entity in extracted_data:
                        source_data = extracted_data[source_entity]
                        target_data = extracted_data[target_entity]
                        
                        # Handle different combinations of single/list entities
                        if isinstance(source_data, list) and isinstance(target_data, list):
                            # Many-to-many relationship (simplistic approach - link all to all)
                            for i in range(len(source_data)):
                                for j in range(len(target_data)):
                                    self._create_relationship(
                                        session, document_id, f"{source_entity}_{i}", 
                                        f"{target_entity}_{j}", relationship_type
                                    )
                        elif isinstance(source_data, list):
                            # Many-to-one relationship
                            for i in range(len(source_data)):
                                self._create_relationship(
                                    session, document_id, f"{source_entity}_{i}", 
                                    target_entity, relationship_type
                                )
                        elif isinstance(target_data, list):
                            # One-to-many relationship
                            for j in range(len(target_data)):
                                self._create_relationship(
                                    session, document_id, source_entity, 
                                    f"{target_entity}_{j}", relationship_type
                                )
                        else:
                            # One-to-one relationship
                            self._create_relationship(
                                session, document_id, source_entity, 
                                target_entity, relationship_type
                            )

    def _create_entity_node(self, session, document_id: str, entity_type: str, entity_data: Dict, node_id: str) -> None:
        """Create a node for an entity with dynamic properties"""
        # Normalize entity data
        normalized_data = {k: self.normalize_value(v) for k, v in entity_data.items() if v is not None}
        normalized_data['node_id'] = node_id
        
        # Sanitize entity type for Neo4j label (remove special characters and spaces)
        sanitized_type = re.sub(r'[^a-zA-Z0-9_]', '_', entity_type)
        
        # Create node with sanitized label
        cypher = f"""
        MATCH (d:Document {{id: $document_id}})
        CREATE (e:{sanitized_type} $properties)
        CREATE (d)-[:CONTAINS]->(e)
        """
        session.run(cypher, document_id=document_id, properties=normalized_data)

    def _create_relationship(self, session, document_id: str, source_id: str, target_id: str, relationship_type: str) -> None:
        """Create relationship between entities"""
        cypher = f"""
        MATCH (d:Document {{id: $document_id}})
        MATCH (source {{node_id: $source_id}})<-[:CONTAINS]-(d)
        MATCH (target {{node_id: $target_id}})<-[:CONTAINS]-(d)
        CREATE (source)-[:{relationship_type}]->(target)
        """
        session.run(cypher, document_id=document_id, source_id=source_id, target_id=target_id)

    def semantic_search_graph(self, query: str, document_type: Optional[str] = None) -> List[Dict]:
        """Perform semantic search on the knowledge graph"""
        # First, ask Gemini to interpret what entities and relationships to search for
        search_prompt = f"""
        Given this search query: "{query}"
        
        Identify:
        1. The main entity types being searched for
        2. Any specific attributes or conditions mentioned
        3. Any relationships that should be traversed
        
        Provide a response in JSON format:
        {{
            "entity_types": ["type1", "type2"],
            "attributes": [{{"entity": "entity_name", "attribute": "attribute_name", "condition": "condition", "value": "value"}}],
            "relationships": [{{"from": "entity_type1", "to": "entity_type2", "type": "relationship_type"}}]
        }}
        """
        
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": search_prompt}]
            )
            # Extract JSON from the response
            json_text = response.choices[0].message.content
            if '```json' in json_text:
                json_text = json_text.split('```json')[1].split('```')[0].strip()
            elif '```' in json_text:
                json_text = json_text.split('```')[1].split('```')[0].strip()
                
            search_params = json.loads(json_text)
        except Exception as e:
            search_params = {"entity_types": [], "attributes": [], "relationships": []}
        
        # Build Cypher query based on search parameters
        cypher_parts = ["MATCH (d:Document)"]
        
        if document_type:
            cypher_parts.append(f"WHERE d.type = '{document_type}'")
        
        # Add entity type matches
        for i, entity_type in enumerate(search_params.get('entity_types', [])):
            var_name = f"e{i}"
            cypher_parts.append(f"MATCH (d)-[:CONTAINS]->({var_name}:{entity_type})")
        
        # Add attribute conditions
        where_conditions = []
        for attr in search_params.get('attributes', []):
            entity_var = attr['entity']
            attribute_name = attr['attribute']
            condition = attr['condition']
            value = attr['value']
            
            if condition == "equals":
                where_conditions.append(f"{entity_var}.{attribute_name} = '{value}'")
            elif condition == "contains":
                where_conditions.append(f"{entity_var}.{attribute_name} CONTAINS '{value}'")
            elif condition == "greater_than":
                where_conditions.append(f"{entity_var}.{attribute_name} > {value}")
            elif condition == "less_than":
                where_conditions.append(f"{entity_var}.{attribute_name} < {value}")
        
        if where_conditions:
            cypher_parts.append("WHERE " + " AND ".join(where_conditions))
        
        # Add relationship traversals
        for rel in search_params.get('relationships', []):
            from_type = rel['from']
            to_type = rel['to']
            rel_type = rel['type'].upper().replace(' ', '_')
            
            cypher_parts.append(f"MATCH ({from_type})-[:{rel_type}]->({to_type})")
        
        # Return results
        return_vars = ["d.id as document_id", "d.type as document_type"]
        for i, entity_type in enumerate(search_params.get('entity_types', [])):
            var_name = f"e{i}"
            return_vars.append(f"{var_name} as {entity_type}")
        
        cypher_parts.append("RETURN " + ", ".join(return_vars))
        
        # Execute query
        cypher_query = " ".join(cypher_parts)
        
        with neo4j_driver.session() as session:
            result = session.run(cypher_query)
            search_results = [dict(record) for record in result]
            
        return search_results

    def process_document(self, input_data: Union[str, bytes], input_type: str) -> Dict:
        """
        Main function to process a document dynamically
        """
        try:
            # Step 1: Upload PDF to S3 and get the content
            if input_type == 'pdf':
                file_name = f"documents/{str(uuid.uuid4())}.pdf"
                
                if S3_BUCKET_NAME:
                    with open(input_data, 'rb') as pdf_file:
                        s3_client.upload_fileobj(pdf_file, S3_BUCKET_NAME, file_name)
                    
                    # Get the content from S3
                    response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=file_name)
                    pdf_content = response['Body'].read()
                    
                    # Extract text from PDF content
                    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                        temp_file.write(pdf_content)
                        temp_file_path = temp_file.name
                else:
                    # Skip S3 and use local file directly for local testing
                    temp_file_path = input_data
                    file_name = "local-file-bypass-s3"

                text = ""
                with fitz.open(temp_file_path) as doc:
                    for page in doc:
                        text += page.get_text()
                
                if S3_BUCKET_NAME:
                    os.unlink(temp_file_path)
                
                if not text:
                    return {"error": "Failed to extract text from PDF"}
                
                # Step 2: Ask Gemini to analyze document
                analysis_prompt = f"""
                Analyze this document and create a knowledge graph structure.
                
                Document Text: {text[:4000]}
                Document S3 Key: {file_name}
                
                Create a knowledge graph structure that captures all important entities and their relationships.
                Focus on extracting invoice-specific information like invoice number, date, items, amounts, etc.
                
                Return ONLY the JSON response in this exact format:
                {{
                    "document_node": {{
                        "type": "Invoice",
                        "properties": {{
                            "invoice_number": "extracted invoice number",
                            "date": "extracted date",
                            "total_amount": "extracted total",
                            "s3_key": "{file_name}"
                        }}
                    }},
                    "entities": [
                        {{
                            "label": "LineItem",
                            "properties": {{
                                "description": "item description",
                                "quantity": "item quantity",
                                "unit_price": "price per unit",
                                "total": "line total"
                            }}
                        }}
                    ],
                    "relationships": [
                        {{
                            "from_node": 0,
                            "to_node": 1,
                            "type": "CONTAINS",
                            "properties": {{}}
                        }}
                    ]
                }}
                """
                
                response = openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": analysis_prompt}]
                )
                response_text = response.choices[0].message.content.strip()
                
                # Clean up the response text to ensure valid JSON
                if '```json' in response_text:
                    response_text = response_text.split('```json')[1].split('```')[0].strip()
                elif '```' in response_text:
                    response_text = response_text.split('```')[1].split('```')[0].strip()
                
                try:
                    graph_structure = json.loads(response_text)
                except json.JSONDecodeError as e:
                    print(f"Failed to parse JSON: {response_text}")
                    return {"error": f"Failed to parse Gemini response: {str(e)}"}
                
                # Step 3: Create knowledge graph
                with neo4j_driver.session() as session:
                    # Create document node
                    doc_props = graph_structure["document_node"]["properties"]
                    doc_type = graph_structure["document_node"]["type"]
                    document_id = str(uuid.uuid4())
                    
                    session.run(
                        """
                        CREATE (d:Invoice $properties)
                        SET d.id = $document_id
                        """,
                        properties=doc_props,
                        document_id=document_id
                    )
                    
                    # Create entity nodes
                    for idx, entity in enumerate(graph_structure["entities"]):
                        sanitized_label = re.sub(r'[^a-zA-Z0-9_]', '_', entity["label"])
                        session.run(
                            f"""
                            MATCH (d:Invoice {{id: $document_id}})
                            CREATE (e:{sanitized_label} $properties)
                            CREATE (d)-[:CONTAINS]->(e)
                            SET e.node_id = $node_id
                            """,
                            document_id=document_id,
                            properties=entity["properties"],
                            node_id=f"node_{idx}"
                        )
                    
                    # Create relationships
                    for rel in graph_structure.get("relationships", []):
                        from_id = f"node_{rel['from_node']}"
                        to_id = f"node_{rel['to_node']}"
                        rel_type = re.sub(r'[^a-zA-Z0-9_]', '_', rel["type"].upper())
                        
                        session.run(
                            f"""
                            MATCH (d:Invoice {{id: $document_id}})
                            MATCH (source {{node_id: $from_id}})<-[:CONTAINS]-(d)
                            MATCH (target {{node_id: $to_id}})<-[:CONTAINS]-(d)
                            CREATE (source)-[r:{rel_type}]->(target)
                            SET r = $properties
                            """,
                            document_id=document_id,
                            from_id=from_id,
                            to_id=to_id,
                            properties=rel.get("properties", {})
                        )
                
                return {
                    "document_id": document_id,
                    "s3_key": file_name,
                    "graph_structure": graph_structure
                }
            
            else:
                return {"error": "Only PDF processing is supported"}
            
        except Exception as e:
            return {"error": str(e)}

    def get_document(self, document_id: str) -> Dict:
        """Retrieve a specific document with all its entities and relationships"""
        with neo4j_driver.session() as session:
            # Get document with all its properties and related items
            result = session.run(
                """
                MATCH (d:Invoice {id: $document_id})
                OPTIONAL MATCH (d)-[:CONTAINS]->(item)
                RETURN d.id as document_id,
                       d.invoice_number as invoice_number,
                       d.date as date,
                       d.total_amount as total_amount,
                       d.s3_key as s3_key,
                       collect(properties(item)) as line_items
                """,
                document_id=document_id
            )
            
            record = result.single()
            if not record:
                return {"error": f"Document {document_id} not found"}
            
            return dict(record)

class DocumentExtractorAPI:
    def __init__(self):
        self.extractor = DynamicDocumentExtractor()
        
    def process_input(self, input_data: Union[str, bytes], input_type: str) -> Dict:
        """API endpoint to process document input"""
        return self.extractor.process_document(input_data, input_type)
        
    def search(self, query: str, document_type: Optional[str] = None) -> List[Dict]:
        """API endpoint to search for information"""
        cypher_query = """
        MATCH (i:Invoice)-[:CONTAINS]->(item)
        WHERE i.invoice_number IS NOT NULL
        RETURN i.id as document_id, 
               i.invoice_number as invoice_number,
               i.date as date,
               i.total_amount as total_amount,
               collect(properties(item)) as line_items
        """
        
        with neo4j_driver.session() as session:
            result = session.run(cypher_query)
            return [dict(record) for record in result]
        
    def get_document(self, document_id: str) -> Dict:
        """API endpoint to retrieve a specific document"""
        return self.extractor.get_document(document_id)

# Example usage
if __name__ == "__main__":
    api = DocumentExtractorAPI()
    
    # Example: Process a local PDF file
    pdf_path = "/Users/pi-in-140/Downloads/Invoice-989B3CF6-0013.pdf"
    result = api.process_input(pdf_path, "pdf")
    print(json.dumps(result, indent=2))
    
    # Example: Perform a search
    search_results = api.search("Find all items with quantity greater than 10")
    print(json.dumps(search_results, indent=2))
    
    # Example: Get document details
    if result.get("document_id"):
        document = api.get_document(result["document_id"])
        print(json.dumps(document, indent=2))