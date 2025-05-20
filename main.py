from fastapi import FastAPI
from contextlib import asynccontextmanager
import asyncio
import json
from config import get_logger, AI_RESPONSE_TOPIC
from database import Database
from kafka_client import KafkaClient
from llm_service import LLMService

logger = get_logger(__name__)

# Load system prompt
with open('system_prompt.txt', 'r') as file:
    SYSTEM_PROMPT = file.read()

# Initialize services
db = Database()
kafka = KafkaClient()
llm_service = LLMService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.check_connection()  # Check MongoDB connection during startup
    kafka.setup_consumer()
    asyncio.create_task(consume_messages())
    yield
    kafka.close()

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
    # Parse the message object
    message_decoded = message.value().decode('utf-8')
    message_value = json.loads(message_decoded)
    msg = message_value['message']
    conversation_id = message_value['conversation_id']
    full_message = "" # Use this to track the full message to be saved to MongoDB
    logger.info(f"Received message from Kafka: |{conversation_id}| {msg}")

    try:
        # Get context and chat history
        context = await db.get_context(conversation_id)
        chat_history = await db.get_history(conversation_id)
    except Exception as e:
        logger.error(f"Error retrieving context or history for conversation {conversation_id}: {e}")
        return

    try:
        response_stream = await llm_service.process_message(msg, context, chat_history, SYSTEM_PROMPT)

        for chunk in response_stream:
            chunk_text = chunk.text()
            full_message += chunk_text
            small_chunk = {
                **message_value,
                "message": chunk_text,
                "last_message": False,
                "error": False,
                "sender": "AIMessage"
            }
            kafka.produce_message(AI_RESPONSE_TOPIC, conversation_id, small_chunk)
            logger.debug(f"Processed chunk: {chunk_text}")

    except Exception as e:
        logger.error(f"Error streaming LLM response: {e}")
        error_chunk = {
            **message_value,
            "message": "",
            "last_message": True,
            "error": True,
            "sender": "AIMessage"
        }
        kafka.produce_error_message(AI_RESPONSE_TOPIC, conversation_id, error_chunk)
        return

    # Send final empty message signaling "done"
    final_chunk = {
        **message_value,
        "message": "",
        "last_message": True,
        "error": False,
        "sender": "AIMessage"
    }

    try:
        kafka.produce_message(AI_RESPONSE_TOPIC, conversation_id, final_chunk)
        logger.debug("Queued final chunk to Kafka")
    except Exception as e:
        logger.error(f"Error sending final message to Kafka: {e}")

    try:
        await db.save_ai_message(conversation_id=conversation_id, message=full_message)
        logger.info(f"Message saved to DB for conversation {conversation_id}")
    except Exception as e:
        logger.error(f"Error saving AI message to DB: {e}")

async def consume_messages():
    while True:
        try:
            msg = kafka.poll_message()
            if msg is not None:
                # Add timeout to process_message to prevent worker hanging
                try:
                    await asyncio.wait_for(process_message(msg), timeout=100)
                except asyncio.TimeoutError:
                    logger.error("Message processing timed out after 100 seconds")
                    # Send error message to Kafka
                    try:
                        message_value = json.loads(msg.value().decode('utf-8'))
                        error_chunk = {
                            **message_value,
                            "message": "Request timed out. Please try again.",
                            "last_message": True,
                            "error": True,
                            "sender": "AIMessage"
                        }
                        kafka.produce_error_message(AI_RESPONSE_TOPIC, message_value['conversation_id'], error_chunk)
                    except Exception as e:
                        logger.error(f"Failed to send timeout error message: {e}")
            else:
                # If no message, yield control back to event loop
                await asyncio.sleep(0.01)  # Small sleep to prevent tight loop
        except Exception as e:
            logger.error(f"Error in message consumption: {e}")
            await asyncio.sleep(1)  # Add delay to prevent tight loop on errors
