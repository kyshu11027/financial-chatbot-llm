from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient as QdrantHttpClient
from langchain_core.tools import tool
from config import get_logger, QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION_NAME, OPENAI_EMBEDDINGS_MODEL_NAME
from typing import Optional, List
import json
from pydantic import BaseModel, Field
from qdrant_client.http import models
from datetime import datetime, timedelta

logger = get_logger(__name__)

class QdrantClient:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(QdrantClient, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize the Qdrant client with configuration"""
        self.client = QdrantHttpClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY
        )
        self.embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDINGS_MODEL_NAME)
        logger.info("Qdrant client initialized")

    def get_client(self):
        """Get the Qdrant client instance"""
        return self.client

    def get_embeddings(self):
        """Get the embeddings model instance"""
        return self.embeddings

class RetrievalIntent(BaseModel):
    user_id: str = Field(description="The ID of the user whose transactions to retrieve")
    num_transactions: int = Field(description="Number of transactions to retrieve")
    time_period_days: Optional[int] = Field(default=None)
    search_query: str = Field(description="Semantic search query")

# Initialize the singleton client
qdrant_client = QdrantClient()

@tool
def retrieve_transactions(intent: RetrievalIntent) -> str:
    """Retrieve relevant transactions from the database based on search intent.
    
    Args:
        intent: RetrievalIntent object specifying search parameters
    
    Returns:
        JSON string of retrieved transactions
    """
    try:
        logger.info(f"Starting transaction retrieval for user_id: {intent.user_id}")
        logger.debug(f"Search parameters: {intent.dict()}")
        
        # Security check - ensure user_id is provided
        if not intent.user_id:
            logger.error("Security violation: user_id not provided")
            return "Error: User ID is required for security"
            
        # Get the singleton client instance
        client = qdrant_client.get_client()
        logger.debug("Retrieved Qdrant client instance")
        
        # Build search query
        search_params = models.SearchParams(
            hnsw_ef=128,
            exact=False
        )
        logger.debug(f"Search parameters configured: {search_params}")
        
        # Always enforce user_id filter - this is a security requirement
        filter_conditions: list[models.Condition] = [
            models.FieldCondition(
                key="metadata.user_id",
                match=models.MatchValue(
                    value=intent.user_id
                )
            )
        ]
        logger.debug("Added user_id filter condition")
        
        # Add additional filters if specified
        if intent.time_period_days:
            start_date = datetime.now() - timedelta(days=intent.time_period_days)
            filter_conditions.append(
                models.FieldCondition(
                    key="metadata.date",
                    range=models.Range(
                        gte=int(start_date.timestamp())
                    )
                )
            )
            logger.debug(f"Added time period filter: {intent.time_period_days} days")
            

            
        
        # Combine filters - user_id filter is always included
        search_filter = models.Filter(
            must=filter_conditions
        )
        logger.debug(f"Final search filter: {search_filter}")
        
        # Get embeddings for the search query
        logger.info("Generating embeddings for search query")
        embeddings = qdrant_client.get_embeddings()
        query_vector = embeddings.embed_query(intent.search_query)
        logger.debug("Embeddings generated successfully")
        
        # Perform search
        logger.info(f"Executing vector search with limit: {intent.num_transactions}")
        logger.debug(f"Query vector shape: {len(query_vector)}")
        logger.debug(f"Search filter: {search_filter}")
        
        search_result = client.query_points(
            collection_name=QDRANT_COLLECTION_NAME,
            query=query_vector,  # This is required for vector search
            limit=intent.num_transactions,
            search_params=search_params,
            query_filter=search_filter
        ).points
        
        logger.info(f"Search completed. Found {len(search_result)} results")
        if len(search_result) == 0:
            logger.warning("No results found. Checking collection stats...")
            try:
                collection_info = client.get_collection(QDRANT_COLLECTION_NAME)
                logger.info(f"Collection info: {collection_info}")
                
                # Check if there are any points in the collection
                collection_stats = client.get_collection(QDRANT_COLLECTION_NAME).points_count
                logger.info(f"Total points in collection: {collection_stats}")
                
                # Get a sample of points to inspect their structure
                sample_points = client.scroll(
                    collection_name=QDRANT_COLLECTION_NAME,
                    limit=5
                )[0]
                logger.info("Sample points from collection:")
                for point in sample_points:
                    logger.info(f"Point ID: {point.id}")
                    logger.info(f"Point payload: {point.payload}")
                    logger.info("---")
                
                # Try a direct search with just the user_id filter
                simple_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.user_id",
                            match=models.MatchValue(
                                value=intent.user_id
                            )
                        )
                    ]
                )
                logger.info(f"Trying simple filter: {simple_filter}")
                simple_search = client.search(
                    collection_name=QDRANT_COLLECTION_NAME,
                    query_vector=query_vector,
                    limit=1,
                    query_filter=simple_filter
                )
                logger.info(f"Simple search found {len(simple_search)} results")
                if simple_search:
                    logger.info(f"Simple search sample: {simple_search[0].payload}")
                    
            except Exception as e:
                logger.error(f"Error checking collection stats: {str(e)}", exc_info=True)
        
        # Convert results to transaction format
        transactions = []
        skipped_count = 0
        for hit in search_result:
            transaction = hit.payload
            metadata = transaction['metadata'] if transaction else {}
            # Additional security check - verify user_id in returned data
            if transaction and metadata['user_id'] == intent.user_id:
                transactions.append(transaction['page_content'])
            else:
                skipped_count += 1
                logger.warning(f"Security check: Skipping transaction with mismatched user_id. Expected: {intent.user_id}, Got: {transaction.get('user_id') if transaction else 'None'}")
        
        if skipped_count > 0:
            logger.warning(f"Skipped {skipped_count} transactions due to user_id mismatch")
        
        logger.info(f"Successfully processed {len(transactions)} transactions")
        return json.dumps(transactions)
        
    except Exception as e:
        logger.error(f"Error retrieving transactions: {str(e)}", exc_info=True)
        return f"Error retrieving transactions: {str(e)}"