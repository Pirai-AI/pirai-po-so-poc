from fastapi import FastAPI, UploadFile, Request, Depends, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import Optional, Dict, List
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv
from pydantic import BaseModel
import json
import logging
from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

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

# Initialize OpenAI
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Neo4j Configuration
NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
NEO4J_USER = os.getenv('NEO4J_USER', 'neo4j')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', 'postgres')


# Pydantic models
class SearchRequest(BaseModel):
    query: str
    params: Optional[Dict] = None
    context: Optional[str] = None


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
async def process_document(file: UploadFile):
    """
    Endpoint to process a document and create knowledge graph
    """
    try:
        # Save the uploaded file temporarily
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Process the document
        result = doc_api.process_input(temp_path, "pdf")

        # Clean up temporary file
        os.remove(temp_path)

        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.post("/search-graph", response_model=AgentResponse)
async def search_graph(
        request: SearchRequest,
        driver: GraphDatabase.driver = Depends(get_neo4j_driver)
):
    """
    AI Agent endpoint to search the knowledge graph using natural language query
    Input:
    - query: Natural language query
    - params: Optional parameters to inject into the query
    - context: Optional context for the agent
    """
    try:
        # Get schema information for better context
        schema = get_schema(driver)
        schema_str = json.dumps(schema, indent=2)

        # Log the retrieved schema information
        logger.info("Retrieved schema information")

        # Initialize LangChain OpenAI chat model
        chat_model = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)

        # Build prompt
        prompt = f"""
        You are a Neo4j query agent that converts natural language to Cypher.

        # Database Schema:
        ```json
        {schema_str}
        ```

        # User Context:
        {request.context or "No additional context provided."}

        # Guidelines for Query Generation:
        1. Analyze the natural language query and identify the entities, relationships, and conditions
        2. Map these to the actual node labels, relationship types, and properties in the schema
        3. Construct a valid Cypher query that correctly represents the user's intent
        4. For date queries, remember that dates are stored as strings in 'YYYY-MM-DD' or 'Month DD, YYYY' format
        5. Include appropriate RETURN clauses to provide comprehensive results
        6. Use parameters for values where appropriate
        7. Include appropriate sorting, filtering, and pagination where relevant
        8. Return related entities for better context
        9. IMPORTANT: When returning properties from nodes, always return them directly (e.g., `n.property`) and NOT as maps/objects
        10. NEVER return objects or maps - always return primitive types (strings, numbers) directly

        # Your Task:
        - Generate a valid Cypher query for: "{request.query}"
        - Provide a brief explanation of how the query works
        - Rate your confidence in the query from 0.0 to 1.0

        Format your response as JSON:
        {{
            "cypher_query": "YOUR CYPHER QUERY HERE",
            "explanation": "Brief explanation of the query",
            "confidence": 0.95
        }}
        """

        # Create messages
        messages = [HumanMessage(content=prompt)]

        # Get response and count tokens
        response = chat_model.invoke(messages)
        response_text = response.content.strip()

        # Update token counter (estimated since exact counts might not always be returned)
        # Using rough estimation: 1 token ≈ 4 characters
        prompt_tokens = len(prompt) // 4
        completion_tokens = len(response_text) // 4
        token_counter.update(prompt_tokens, completion_tokens)

        # Extract JSON from response
        try:
            # Try to parse directly
            agent_response = json.loads(response_text)
        except json.JSONDecodeError:
            # If direct parsing fails, try to extract JSON block
            try:
                if "```json" in response_text:
                    json_part = response_text.split("```json")[1].split("```")[0].strip()
                elif "```" in response_text:
                    json_part = response_text.split("```")[1].split("```")[0].strip()
                else:
                    # Try to find JSON-like content
                    start_idx = response_text.find('{')
                    end_idx = response_text.rfind('}') + 1
                    if start_idx >= 0 and end_idx > start_idx:
                        json_part = response_text[start_idx:end_idx]
                    else:
                        raise ValueError("Could not extract JSON from response")

                agent_response = json.loads(json_part)
            except Exception as json_ex:
                logger.error(f"Failed to parse Gemini response: {str(json_ex)}")
                # Fallback to a default structure
                agent_response = {
                    "cypher_query": "Could not generate valid Cypher query",
                    "explanation": "Failed to parse the model response",
                    "confidence": 0.0
                }

        # Add token counts to the response
        agent_response['token_counts'] = token_counter.get_counts()

        cypher_query = agent_response.get("cypher_query", "")
        explanation = agent_response.get("explanation", "")
        confidence = agent_response.get("confidence", 0.5)

        logger.info(f"Generated Cypher query with confidence {confidence}")

        # Replace parameters if provided
        if request.params:
            # For simple parameter replacement in string
            for param, value in request.params.items():
                param_placeholder = f"${param}"
                if isinstance(value, str):
                    cypher_query = cypher_query.replace(param_placeholder, f"'{value}'")
                else:
                    cypher_query = cypher_query.replace(param_placeholder, str(value))

        # Execute the Cypher query
        try:
            logger.info(f"Executing Cypher query: {cypher_query}")
            with driver.session() as session:
                result = session.run(cypher_query, request.params or {})
                records = [dict(record) for record in result]

                # Convert Neo4j types to JSON-serializable types
                processed_records = process_neo4j_results(records)

                return AgentResponse(
                    query=request.query,
                    cypher_query=cypher_query,
                    results=processed_records,
                    total_results=len(processed_records),
                    explanation=explanation,
                    query_confidence=confidence,
                    token_counts=token_counter.get_counts()
                )

        except Exception as db_error:
            logger.error(f"Database error executing query: {str(db_error)}")
            # Generate a fallback query if possible
            fallback_query = generate_fallback_query(request.query, schema)

            with driver.session() as session:
                result = session.run(fallback_query)
                records = [dict(record) for record in result]
                processed_records = process_neo4j_results(records)

                return AgentResponse(
                    query=request.query,
                    cypher_query=fallback_query,
                    results=processed_records,
                    total_results=len(processed_records),
                    explanation=f"Original query failed with error: {str(db_error)}. Used fallback query.",
                    query_confidence=0.3,
                    token_counts=token_counter.get_counts()
                )

    except Exception as e:
        logger.error(f"General error in search_graph: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error in AI agent: {str(e)}"
        )


def process_neo4j_results(records):
    """Process Neo4j results to make them JSON serializable"""
    processed_records = []
    for record in records:
        processed_record = {}
        for key, value in record.items():
            # Directly serialize the value without creating nested structures
            processed_record[key] = serialize_neo4j_value(value, flatten=True)
        processed_records.append(processed_record)
    return processed_records


def serialize_neo4j_value(value, flatten=False):
    """
    Serialize a Neo4j value to a JSON-compatible format
    Args:
        value: The value to serialize
        flatten: If True, flattens node and relationship properties into primitive values
    """
    if hasattr(value, 'items') and callable(value.items):  # Neo4j Node or Relationship
        if flatten:
            # For nodes and relationships, just return their properties as primitive values
            return {k: serialize_neo4j_value(v, flatten=True) for k, v in value.items()}
        else:
            # Only serialize the properties
            properties = {}
            for k, v in value.items():
                properties[k] = serialize_neo4j_value(v, flatten=True)
            return properties
    elif isinstance(value, list):
        return [serialize_neo4j_value(item, flatten=True) for item in value]
    elif isinstance(value, dict):
        # For dict, process each key-value pair
        return {k: serialize_neo4j_value(v, flatten=True) for k, v in value.items()}
    elif isinstance(value, (int, float, bool, str, type(None))):
        # Primitive types are fine
        return value
    else:
        # Convert anything else to string
        return str(value)


def generate_fallback_query(query, schema):
    """Generate a fallback query when the main query fails"""
    # Find the main node labels from schema
    node_labels = list(schema["nodes"].keys())

    if not node_labels:
        # If no schema available, use a very basic query
        return "MATCH (n) RETURN n.id as id LIMIT 10"

    # Use the first node label as the primary one for the query
    primary_label = node_labels[0]

    # Get properties for this label
    properties = schema["nodes"].get(primary_label, {}).get("properties", {})
    prop_names = list(properties.keys())

    if not prop_names:
        return f"MATCH (n:{primary_label}) RETURN id(n) as id LIMIT 10"

    # Select some common properties to return, avoid returning objects
    select_props = prop_names[:3]  # Take first 3 properties

    # Build a basic query returning primitive properties only
    select_clause = ", ".join([f'n.{p} as {p}' for p in select_props])
    return f"""
    MATCH (n:{primary_label})
    RETURN {select_clause}
    LIMIT 20
    """


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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)