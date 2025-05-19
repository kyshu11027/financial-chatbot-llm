import os
import logging
from dotenv import load_dotenv

load_dotenv()

# Kafka Configuration
KAFKA_CONFIG = {
    'bootstrap.servers': os.getenv('KAFKA_BOOTSTRAP_SERVERS'),
    'security.protocol': 'SASL_SSL',
    'sasl.mechanisms': 'PLAIN',
    'sasl.username': os.getenv('KAFKA_API_KEY'),
    'sasl.password': os.getenv('KAFKA_API_SECRET'),
}

# Kafka Topics
USER_MESSAGE_TOPIC = "user_message"
AI_RESPONSE_TOPIC = "ai_response"
GROUP_ID = "message_consumer"

# MongoDB Configuration
MONGODB_URI = os.getenv("MONGODB_URI")
CONTEXT_COLLECTION_NAME = "contexts"
MESSAGE_COLLECTION_NAME = "messages"

# OpenAI Configuration
OPENAI_KEY = os.getenv('OPENAI_API_KEY')

def setup_logging():
    # Get log level from environment variable, default to INFO
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    if log_level not in ['DEBUG', 'INFO', 'WARNING', 'ERROR']:
        log_level = 'INFO'
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set PyMongo's logger to WARNING level to reduce noise
    logging.getLogger('pymongo').setLevel(logging.WARNING)
    logging.getLogger('pymongo.topology').setLevel(logging.WARNING)
    
    # Set Kafka's logger to WARNING level
    logging.getLogger('confluent_kafka').setLevel(logging.WARNING)
    
    # Set Uvicorn's logger to WARNING level
    logging.getLogger('uvicorn').setLevel(logging.WARNING)
    logging.getLogger('uvicorn.access').setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)

logger = setup_logging() 