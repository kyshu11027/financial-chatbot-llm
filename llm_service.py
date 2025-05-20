from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from config import OPENAI_KEY, OPENAI_MODEL_NAME, get_logger

logger = get_logger(__name__)

class LLMService:
    def __init__(self):
        self.llm = ChatOpenAI(api_key=OPENAI_KEY, streaming=True, temperature=0.7)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "{system_prompt}\n{context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ])

    def get_chain(self, system_prompt):
        return self.prompt | self.llm

    async def process_message(self, message, context, chat_history, system_prompt):
        try:
            chain = self.get_chain(system_prompt)
            response_stream = chain.stream({
                "system_prompt": system_prompt,
                "context": context,
                "chat_history": chat_history,
                "input": message
            })
            return response_stream
        except Exception as e:
            logger.error(f"Error streaming LLM response: {e}")
            raise 