import os
import logging
from dotenv import load_dotenv

load_dotenv()

# Kafka Configuration
KAFKA_CONFIG = {
    'bootstrap.servers': os.getenv('KAFKA_SERVER', ''),
    'security.protocol': 'SASL_SSL',
    'sasl.mechanisms': 'PLAIN',
    'sasl.username': os.getenv('KAFKA_USERNAME', ''),
    'sasl.password': os.getenv('KAFKA_PASSWORD', ''),
}

# Kafka Topics
USER_MESSAGE_TOPIC = "user_message"
AI_RESPONSE_TOPIC = "ai_response"
GROUP_ID = "message_consumer"

# MongoDB Configuration
MONGODB_URI = os.getenv('MONGODB_URI', '')
CONTEXT_COLLECTION_NAME = "contexts"
MESSAGE_COLLECTION_NAME = "messages"

# OpenAI Configuration
OPENAI_KEY = os.getenv('OPENAI_API_KEY', '')
OPENAI_MODEL_NAME = os.getenv('OPENAI_MODEL_NAME', '')
OPENAI_EMBEDDINGS_MODEL_NAME = os.getenv('OPENAI_EMBEDDINGS_MODEL_NAME', '')

# Google Gemini Configuration
GEMINI_KEY = os.getenv('GEMINI_API_KEY', '')
GEMINI_MODEL_NAME = os.getenv('GEMINI_MODEL', '')

# Qdrant Configuration
QDRANT_URL = os.getenv('QDRANT_URL', '')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY', '')
QDRANT_COLLECTION_NAME = "transactions"

def get_logger(name):
    """
    Get a logger instance for a specific module.
    Args:
        name (str): The name of the module (usually __name__)
    Returns:
        logging.Logger: A configured logger instance
    """
    # Get log level from environment variable, default to INFO
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    if log_level not in ['DEBUG', 'INFO', 'WARNING', 'ERROR']:
        log_level = 'INFO'
    
    # Configure root logger if not already configured
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='[%(levelname)s] %(asctime)s |%(name)s| %(message)s'
        )
        
        # Set PyMongo's logger to WARNING level to reduce noise
        logging.getLogger('pymongo').setLevel(logging.WARNING)
        logging.getLogger('pymongo.topology').setLevel(logging.WARNING)
        
        # Set Kafka's logger to WARNING level
        logging.getLogger('confluent_kafka').setLevel(logging.WARNING)
        
        # Set Uvicorn's logger to WARNING level
        logging.getLogger('uvicorn').setLevel(logging.WARNING)
        logging.getLogger('uvicorn.access').setLevel(logging.WARNING)
    
    return logging.getLogger(name)

# Create a logger for the config module itself
logger = get_logger(__name__) 