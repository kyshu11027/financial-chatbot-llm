from fastapi import FastAPI
from confluent_kafka import Consumer, Producer
from contextlib import asynccontextmanager
import asyncio
import os
import json
import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from pymongo import MongoClient, errors
import time
from dotenv import load_dotenv

load_dotenv()
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up constants
OPENAI_KEY = os.getenv('OPENAI_API_KEY')
USER_MESSAGE_TOPIC = "user_message"
AI_RESPONSE_TOPIC = "ai_response"
GROUP_ID = "message_consumer"
CONTEXT_COLLECTION_NAME = "contexts"
MESSAGE_COLLECTION_NAME = "messages" 

# Global variables
producer = Producer({
    'bootstrap.servers': os.getenv('KAFKA_BOOTSTRAP_SERVERS'),
    'security.protocol': 'SASL_SSL',
    'sasl.mechanisms': 'PLAIN',
    'sasl.username': os.getenv('KAFKA_API_KEY'),
    'sasl.password': os.getenv('KAFKA_API_SECRET'),
})


client = MongoClient(os.getenv("MONGODB_URI"), tls=True)
db = client["conversations"]
context_collection = db[CONTEXT_COLLECTION_NAME]
messages_collection = db[MESSAGE_COLLECTION_NAME]

# MongoDB ping check function
async def check_mongo_connection():
    try:
        # Attempt to ping the MongoDB server
        client.admin.command('ping')
        logger.info("MongoDB connection successful!")
    except errors.PyMongoError as e:
        logger.error(f"MongoDB connection failed: {e}")
        raise Exception(f"MongoDB connection failed: {e}")

llm = ChatOpenAI(api_key=OPENAI_KEY, streaming=True, temperature=0.7)

@asynccontextmanager
async def lifespan(app: FastAPI):
    await check_mongo_connection()  # Check MongoDB connection during startup

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

@app.get("/health")
async def health_check():
    return {"status": "healthy"}    

async def process_message(message):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Given the following transaction history and financial information, answer the user's questions or generate insights based on this data:\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),  # Include history here
        ("user", "{input}"),
    ])
    chain = prompt | llm

    message_decoded = message.value().decode('utf-8')
    message_value = json.loads(message_decoded)
    msg = message_value['message']
    conversation_id = message_value['conversation_id']

    full_message = ""

    try:
        # Pull context and chat history from MongoDB
        context = await get_context(conversation_id)
        chat_history = await get_history(conversation_id)
    except Exception as e:
        logger.error(f"Error retrieving context or history for conversation {conversation_id}: {e}")
        return

    try:
        response_stream = chain.stream({
            "context": context,
            "chat_history": chat_history, 
            "input": msg})

        for chunk in response_stream:
            chunk_text = chunk.text()
            full_message += chunk_text
            small_chunk = {
                **message_value,
                "message": chunk_text,
                "last_message": False
            }
            producer.produce(AI_RESPONSE_TOPIC, key=conversation_id, value=json.dumps(small_chunk))
            producer.poll(0)  # Let producer handle delivery in background (non-blocking)

            logger.info(f"Queued chunk to Kafka: {chunk}")

    except Exception as e:
        logger.error(f"Error streaming LLM response: {e}")
        return

    # Send final empty message signaling "done"
    final_chunk = {
        **message_value,
        "message": "",
        "last_message": True
    }

    try:
        producer.produce(AI_RESPONSE_TOPIC, value=json.dumps(final_chunk))
        producer.poll(0)
        logger.info(f"Queued final [DONE] chunk to Kafka.")
    except Exception as e:
        logger.error(f"Error sending final message to Kafka: {e}")

    try:
        # Save the full AI message to MongoDB
        await save_ai_message_to_db(conversation_id=conversation_id, message=full_message)
        logger.info(f"Message saved to DB for conversation {conversation_id}")
    except Exception as e:
        logger.error(f"Error saving AI message to DB: {e}")

async def consume_messages():
    consumer = Consumer({
        'bootstrap.servers': os.getenv('KAFKA_BOOTSTRAP_SERVERS'),
        'security.protocol': 'SASL_SSL',
        'sasl.mechanisms': 'PLAIN',
        'sasl.username': os.getenv('KAFKA_API_KEY'),
        'sasl.password': os.getenv('KAFKA_API_SECRET'),
        'session.timeout.ms': '45000',
        'client.id': 'python-client-1',
        'group.id': GROUP_ID,
        'auto.offset.reset': 'latest',
    })
    consumer.subscribe([USER_MESSAGE_TOPIC])
    logger.info("Kafka consumer started, waiting for messages...")

    while True:
        try:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                logger.error(f"Consumer error: {msg.error()}")
                continue
            await process_message(msg)

        except Exception as e:
            logger.error(f"Error in message consumption: {e}")

async def get_context(conversation_id):
    try:
        context_doc = context_collection.find_one({"conversation_id": conversation_id})
        if not context_doc:
            raise Exception(f"No context found for conversation_id: {conversation_id}")
        
        # Normalize transactions
        transactions = []
        for t in context_doc.get('transactions', []):
            transactions.append({
                'transaction_id': t['transactionid'],
                'date': t['date'],
                'amount': t['amount'],
                'name': t['name'],
                'merchant': t.get('merchantname', ''),
                'categories': t.get('category', []),
                'pending': t['pending'],
            })

        # Format context
        context = f"""
            I make {context_doc['income']} dollars a month. 
            I want to save {context_doc['savings_goal']} a month. 
            Here is a list of transactions I have made in the last 30 days:
        """
        for t in transactions:
            context += f"""
                Transaction {t['transaction_id']}: {t['name']} at {t['merchant']} on {t['date']} for ${t['amount']}. 
                Categories: {', '.join(t['categories'])}. 
                Pending: {t['pending']}\n
            """
        return context
    except Exception as e:
        logger.error(f"Error retrieving context for conversation_id {conversation_id}: {e}")
        raise

async def get_history(conversation_id):
    try:
        chat_history = list(messages_collection.find({"conversation_id": conversation_id}).sort("timestamp", 1))
        if not chat_history:
            raise Exception(f"No chat history found for conversation_id: {conversation_id}")

        formatted_history = []
        for message in chat_history:
            if message['sender'] == "UserMessage":
                formatted_history.append(HumanMessage(content=message['message']))
            else:
                formatted_history.append(AIMessage(content=message['message']))
        return formatted_history
    except Exception as e:
        logger.error(f"Error retrieving history for conversation_id {conversation_id}: {e}")
        raise

async def save_ai_message_to_db(conversation_id, message):
    try:
        messages_collection.insert_one({
            "conversation_id": conversation_id,
            "sender": "AIMessage",
            "message": message,
            "timestamp": int(time.time())
        })
    except Exception as e:
        logger.error(f"Error saving message to MongoDB: {e}")
        raise
