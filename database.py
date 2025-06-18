from pymongo import MongoClient, errors
import certifi
import time
from config import MONGODB_URI, CONTEXT_COLLECTION_NAME, MESSAGE_COLLECTION_NAME, get_logger

logger = get_logger(__name__)

class Database:
    def __init__(self):
        self.client = MongoClient(MONGODB_URI, tls=True, tlsCAFile=certifi.where())
        self.db = self.client["conversations"]
        self.context_collection = self.db[CONTEXT_COLLECTION_NAME]
        self.messages_collection = self.db[MESSAGE_COLLECTION_NAME]

    async def check_connection(self):
        try:
            self.client.admin.command('ping')
            logger.info("MongoDB connection successful!")
        except errors.PyMongoError as e:
            logger.error(f"MongoDB connection failed: {e}")
            raise Exception(f"MongoDB connection failed: {e}")

    async def get_context(self, conversation_id):
        try:
            context_doc = self.context_collection.find_one({"conversation_id": conversation_id})
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
                    'category': t.get('category') or [],
                    'pending': t['pending'],
                })

            accounts = []
            for a in context_doc.get('accounts', []): 
                balance = a.get('balances', {})
                normalized_balance = {
                    'available': balance.get('available', None),
                    'current': balance.get('current', 0.0),
                    'limit': balance.get('limit', None),
                    'iso_currency_code': balance.get('iso_currency_code', ''),
                }

                normalized_account = {
                    'account_id': a.get('account_id', ''),
                    'balances': normalized_balance,
                    'mask': a.get('mask', ''),
                    'name': a.get('name', 'Unnamed Account'),
                    'official_name': a.get('official_name', 'Unnamed Account'),
                    'subtype': a.get('subtype', ''),
                    'type': a.get('type', ''),
                }
                accounts.append(normalized_account)

            # Format context
            context = f"""
                My name is {context_doc['name']}.
                I make {context_doc['income']} dollars a month. 
                I want to save {context_doc['savings_goal']} a month. 
                Here is a list of transactions I have made in the last 30 days:
            """

            for t in transactions:
                context += f"""
                    Transaction {t['transaction_id']}: {t['name']} at {t['merchant']} on {t['date']} for ${t['amount']}. 
                    Category: {t['category']}. 
                    Pending: {t['pending']}\n
                """

            context += "Here is a list of my current account balances:"
            for account in accounts:
                context += f"""
                    {account['official_name']} : {account['balances']['current']} {account['balances']['iso_currency_code']}
                """

            context += "Here is a list of my monthly expenses:"
            for monthly_expense in context_doc['additional_monthly_expenses']:
                context += f"""
                    Name: {monthly_expense['name']}
                    Amount: {monthly_expense['amount']}
                """
                if monthly_expense['description'] != "":
                    context += f'Description: {monthly_expense['description']}'
                    
            return context
        except Exception as e:
            logger.error(f"Error retrieving context for conversation_id {conversation_id}: {e}")
            raise


    async def get_context_no_transactions(self, conversation_id):
        try:
            context_doc = self.context_collection.find_one({"conversation_id": conversation_id})
            if not context_doc:
                raise Exception(f"No context found for conversation_id: {conversation_id}")

            accounts = []
            for a in context_doc.get('accounts', []): 
                balance = a.get('balances', {})
                normalized_balance = {
                    'available': balance.get('available', None),
                    'current': balance.get('current', 0.0),
                    'limit': balance.get('limit', None),
                    'iso_currency_code': balance.get('iso_currency_code', ''),
                }

                normalized_account = {
                    'account_id': a.get('account_id', ''),
                    'balances': normalized_balance,
                    'mask': a.get('mask', ''),
                    'name': a.get('name', 'Unnamed Account'),
                    'official_name': a.get('official_name', 'Unnamed Account'),
                    'subtype': a.get('subtype', ''),
                    'type': a.get('type', ''),
                }
                accounts.append(normalized_account)

            # Format context
            context = f"My name is {context_doc['name']}.\nI make {context_doc['income']} dollars a month.\nI want to save {context_doc['savings_goal']} a month.\n\n"

            context += "Here is a list of my current account balances:\n"
            for account in accounts:
                context += f"{account['official_name']} : {account['balances']['current']} {account['balances']['iso_currency_code']}\n"

            context += "Here is a list of my recurring monthly expenses:\n"
            for monthly_expense in context_doc['additional_monthly_expenses']:
                context += f"Name: {monthly_expense['name']} | Amount: {monthly_expense['amount']}"
                if monthly_expense['description'] != "":
                    context += f' | Description: {monthly_expense['description']}'
                context += "\n"
                    
            return context
        except Exception as e:
            logger.error(f"Error retrieving context for conversation_id {conversation_id}: {e}")
            raise

    async def get_history(self, conversation_id):
        try:
            chat_history = list(self.messages_collection.find({"conversation_id": conversation_id}).sort("timestamp", 1))
            if not chat_history:
                raise Exception(f"No chat history found for conversation_id: {conversation_id}")

            from langchain_core.messages import HumanMessage, AIMessage
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

    async def save_ai_message(self, conversation_id, message):
        try:
            self.messages_collection.insert_one({
                "conversation_id": conversation_id,
                "sender": "AIMessage",
                "message": message,
                "timestamp": int(time.time())
            })
        except Exception as e:
            logger.error(f"Error saving message to MongoDB: {e}")
            raise 