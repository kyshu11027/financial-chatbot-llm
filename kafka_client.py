from confluent_kafka import Consumer, Producer
import json
from config import KAFKA_CONFIG, USER_MESSAGE_TOPIC, GROUP_ID, get_logger

logger = get_logger(__name__)

class KafkaClient:
    def __init__(self):
        self.producer = Producer(KAFKA_CONFIG)
        self.consumer = None

    def setup_consumer(self):
        consumer_config = {
            **KAFKA_CONFIG,
            'session.timeout.ms': '45000',
            'client.id': 'python-client-1',
            'group.id': GROUP_ID,
            'auto.offset.reset': 'latest',
        }
        self.consumer = Consumer(consumer_config)
        self.consumer.subscribe([USER_MESSAGE_TOPIC])
        logger.info("Kafka consumer started, waiting for messages...")

    def produce_message(self, topic, key, value):
        try:
            self.producer.produce(topic, key=key, value=json.dumps(value))
            self.producer.poll(0)  # Non-blocking
            logger.debug(f"Queued message to Kafka topic {topic}")
        except Exception as e:
            logger.error(f"Error producing message to Kafka: {e}")
            raise

    def produce_error_message(self, topic, key, value):
        try:
            self.producer.produce(topic, key=key, value=json.dumps(value))
            self.producer.flush()  # Ensure error message gets delivered
            logger.debug(f"Queued error message to Kafka topic {topic}")
        except Exception as e:
            logger.error(f"Failed to send error message to Kafka: {e}")
            raise

    def poll_message(self):
        if self.consumer is None:
            logger.error("Kafka consumer is not initialized.")
            return None
        try:
            msg = self.consumer.poll(0.1)  # Reduced timeout to 100ms to be more responsive
            if msg is None:
                return None
            if msg.error():
                logger.error(f"Consumer error: {msg.error()}")
                return None
            return msg
        except Exception as e:
            logger.error(f"Error in message consumption: {e}")
            return None

    def close(self):
        if self.consumer:
            self.consumer.close()
        self.producer.flush() 