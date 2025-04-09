from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import logging
from contextlib import asynccontextmanager
from confluent_kafka import Consumer
import asyncio
import httpx
import os
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TOPIC_NAME = "user_message"
GROUP_ID = "message_consumer"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup event
    asyncio.create_task(consume_messages())
    yield  # This will pause here until the app is shutting down
    # Shutdown event can be handled here if needed

app = FastAPI(
    title="Finance Chatbot API",
    description="A FastAPI backend for the finance chatbot application",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
async def root():
    return {"message": "Welcome to the Finance Chatbot API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}


async def process_message(message):
    async with httpx.AsyncClient() as client:
        message_decoded = message.value().decode('utf-8')
        message_value = json.loads(message_decoded)
        message_value['sender'] = 'AIMessage'
        response = await client.post("http://localhost:8080/internal/message/receive", json=message_value, headers={"X-API-Key": os.getenv("INTERNAL_API_KEY")})
        
        logger.info(f"Processed message: {message_decoded}, Response: {response.text}")

async def consume_messages():
    consumer = Consumer({
        'bootstrap.servers': os.getenv('KAFKA_BOOTSTRAP_SERVERS'),
        'security.protocol': 'SASL_SSL',
        'sasl.mechanisms': 'PLAIN',
        'sasl.username': os.getenv('KAFKA_API_KEY'),
        'sasl.password': os.getenv('KAFKA_API_SECRET'),
        'session.timeout.ms':'45000',
        'client.id':'python-client-1',
        'group.id': GROUP_ID,
        'auto.offset.reset': 'earliest',
    })
    consumer.subscribe([TOPIC_NAME])
    logger.info("Kafka consumer started, waiting for messages...")
    while True:
        msg = consumer.poll(1.0)  # Timeout of 1 second
        if msg is None:
            continue
        if msg.error():
            logger.error(f"Error while consuming message: {msg.error()}")
            continue
        await process_message(msg)

