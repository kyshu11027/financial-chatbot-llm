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
    """Intent for retrieving user transactions with specific search criteria."""
    
    user_id: str = Field(
        description="The ID of the user whose transactions to retrieve"
    )
    num_transactions: int = Field(
        default=10,
        description="Number of transactions to retrieve (between 1 and 50)",
        ge=1,
        le=50
    )
    time_period_days: Optional[int] = Field(
        default=None,
        description="Optional: Limit to transactions from the last N days (e.g., 30 for last month, 7 for last week)"
    )
    search_query: str = Field(
        description="Semantic search query describing what transactions to find (e.g., 'monthly spending categories', 'grocery purchases', 'entertainment expenses', 'rent and housing costs')"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": "user123",
                "search_query": "monthly spending categories including rent and groceries",
                "num_transactions": 30,
                "time_period_days": 30
            }
        }

# Updated tool definition with explicit schema

# Initialize the singleton client
qdrant_client = QdrantClient()

@tool(args_schema=RetrievalIntent)
def retrieve_transactions(user_id: str, num_transactions: int, time_period_days: Optional[int], search_query: str) -> List[str]:
    """Retrieve relevant transactions from the database based on search intent.
    
    Args:
        intent: RetrievalIntent object specifying search parameters
    
    Returns:
        JSON string of retrieved transactions
    """
    try:
        logger.info(f"Starting transaction retrieval for user_id: {user_id}")
        
        # Security check - ensure user_id is provided
        if not user_id:
            logger.error("Security violation: user_id not provided")
            return []
            
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
                    value=user_id
                )
            )
        ]
        logger.debug("Added user_id filter condition")
        
        # Add additional filters if specified
        if time_period_days:
            start_date = datetime.now() - timedelta(days=time_period_days)
            filter_conditions.append(
                models.FieldCondition(
                    key="metadata.date",
                    range=models.Range(
                        gte=int(start_date.timestamp())
                    )
                )
            )
            logger.debug(f"Added time period filter: {time_period_days} days")
            
        # Combine filters - user_id filter is always included
        search_filter = models.Filter(
            must=filter_conditions
        )
        logger.debug(f"Final search filter: {search_filter}")
        
        # Get embeddings for the search query
        logger.info("Generating embeddings for search query")
        embeddings = qdrant_client.get_embeddings()
        query_vector = embeddings.embed_query(search_query)
        logger.debug("Embeddings generated successfully")
        
        # Perform search
        logger.info(f"Executing vector search with limit: {num_transactions}")
        logger.debug(f"Query vector shape: {len(query_vector)}")
        logger.debug(f"Search filter: {search_filter}")
        
        search_result = client.query_points(
            collection_name=QDRANT_COLLECTION_NAME,
            query=query_vector,  # This is required for vector search
            limit=num_transactions,
            search_params=search_params,
            query_filter=search_filter
        ).points
        
        logger.info(f"Search completed. Found {len(search_result)} results")
        # Convert results to transaction format
        transactions = []
        skipped_count = 0
        for hit in search_result:
            transaction = hit.payload
            metadata = transaction['metadata'] if transaction else {}
            # Additional security check - verify user_id in returned data
            if transaction and metadata['user_id'] == user_id:
                transactions.append(transaction['page_content'])
            else:
                skipped_count += 1
                logger.warning(f"Security check: Skipping transaction with mismatched user_id. Expected: {user_id}, Got: {transaction.get('user_id') if transaction else 'None'}")
        
        if skipped_count > 0:
            logger.warning(f"Skipped {skipped_count} transactions due to user_id mismatch")
        
        logger.info(f"Successfully processed {len(transactions)} transactions")
        return transactions
        
    except Exception as e:
        logger.error(f"Error retrieving transactions: {str(e)}", exc_info=True)
        return []