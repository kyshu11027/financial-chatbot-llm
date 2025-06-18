from typing import TypedDict, List, Optional, Literal
from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage, ToolCall, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from tools.qdrant_tool import retrieve_transactions
from config import GEMINI_KEY, GEMINI_MODEL_NAME, get_logger
from collections import deque
from datetime import date

logger = get_logger(__name__)

# Load prompts
with open('system_prompt.txt', 'r') as file:
    SYSTEM_PROMPT = file.read()

with open('tool_prompt.txt', 'r') as file:
    TOOL_PROMPT = file.read()

# Simple state structure
class AgentState(TypedDict):
    user_query: str
    user_id: str
    user_context: str
    chat_history: List[BaseMessage]
    tool_calls: deque[ToolCall]
    retrieved_transactions: List[str]
    final_response: Optional[str]

class LLMAgent:
    def __init__(self):
        
        # LLM with tool calling for retrieval decisions
        self.tool_llm = ChatGoogleGenerativeAI(
            api_key=GEMINI_KEY, 
            model=GEMINI_MODEL_NAME, 
            temperature=0.5,
        ).bind_tools([retrieve_transactions])
        
        # LLM for final response generation
        self.response_llm = ChatGoogleGenerativeAI(
            api_key=GEMINI_KEY, 
            model=GEMINI_MODEL_NAME, 
            temperature=0.5,
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "{system_prompt}\n{context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ])
        
        # Build the state graph
        self.graph = self._build_graph()
        logger.info("Agent initialized with state graph")
    
    def _build_graph(self):
        """Build a simple 3-node state graph"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("decide_retrieval", self._decide_retrieval_node)
        workflow.add_node("retrieve_data", self._retrieve_data_node)
        workflow.add_node("generate_response", self._generate_response_node)
        
        # Add edges
        workflow.set_entry_point("decide_retrieval")
        workflow.add_conditional_edges(
            "decide_retrieval",
            self._should_retrieve,
            {
                "retrieve": "retrieve_data",
                "respond": "generate_response"
            }
        )
        workflow.add_edge("retrieve_data", "generate_response")
        workflow.add_edge("generate_response", END)
        
        return workflow.compile()
    
    def _decide_retrieval_node(self, state: AgentState) -> AgentState:
        """Node 1: Let LLM decide if transaction retrieval is needed"""
        logger.info("Deciding if transaction retrieval is needed")
        
        system_prompt = f"The current date is {date.today().isoformat()}.\n{TOOL_PROMPT}"
        
        chain = self.get_tool_chain()
        response = chain.invoke({
            "system_prompt": system_prompt,
            "context": state['user_context'],
            "chat_history": state['chat_history'],
            "input": state['user_query']
        })
        
        logger.info(f"Decide Retrieval Response: {response}")

        # Check if LLM wants to retrieve transactions
        if hasattr(response, 'tool_calls') and isinstance(response, AIMessage) and response.tool_calls:
            # Store the tool call arguments for the next node
            tool_call = response.tool_calls[0]  # Assume first tool call
            state['tool_calls'].append(tool_call)
            logger.info(f"LLM requested retrieval with args: {tool_call['args']}")
        else:
            logger.info("LLM decided no retrieval needed")
        
        return state
    
    def _retrieve_data_node(self, state: AgentState) -> AgentState:
        """Node 2: Execute transaction retrieval"""
        logger.info("Retrieving transaction data")

        if len(state['tool_calls']) == 0:
            return state  # No tool calls, skip retrieval
        
        try:
            tool_call = state['tool_calls'].popleft()
            # Get tool arguments from previous node and add user_id
            
            tool_args = tool_call['args'].copy()
            tool_args['user_id'] = state['user_id']

            logger.debug(f"Tool call arguments: {tool_args}")
            
            # Execute retrieval
            transactions = retrieve_transactions.invoke(tool_args)
            state['retrieved_transactions'] = transactions
            logger.info(f"Retrieved {len(transactions)} transactions")
            
        except Exception as e:
            logger.error(f"Error retrieving transactions: {str(e)}")
            state['retrieved_transactions'] = [f"Error: {str(e)}"]
        
        return state
    
    def _generate_response_node(self, state: AgentState) -> AgentState:
        """Node 3: Generate final response with retrieved data"""
        logger.info("Generating final response")
        
        # Build context from retrieved transactions
        context = f"{state['user_context']}\n" 
        if state['retrieved_transactions']:
            context += "Retrieved Transaction Data:\n" + "\n".join(state['retrieved_transactions'])

        logger.debug(f"Context for response: {context}")
        
        system_prompt = f"The current date is {date.today().isoformat()}.\n\n{SYSTEM_PROMPT}"
        
        chain = self.get_response_chain()
        response = chain.invoke({
            "system_prompt": system_prompt,
            "context": context,
            "chat_history": state['chat_history'],
            "input": state['user_query']
        })
        state['final_response'] = response.text()
        
        logger.info("Final response generated")
        return state
    
    def _should_retrieve(self, state: AgentState) -> Literal["retrieve", "respond"]:
        """Routing function: decide whether to retrieve data or go straight to response"""
        if len(state['tool_calls']) > 0:
            logger.info("Routing to retrieve_data")
            return "retrieve"
        else:
            logger.info("Routing to generate_response")
            return "respond"
        
    def get_response_chain(self):
        return self.prompt | self.response_llm
    
    def get_tool_chain(self):
        return self.prompt | self.tool_llm
    
    async def query(self, user_query: str, user_id: str, user_context: str = "", chat_history: List = []) -> dict:
        """Process a user query through the state graph"""
        logger.info(f"Processing query for user {user_id}: {user_query}")
        
        initial_state = {
            'user_query': user_query,
            'user_id': user_id,
            'tool_calls': deque(),
            'user_context': user_context,
            'chat_history': chat_history,
            'retrieved_transactions': [],
            'final_response': None
        }
        
        # Run the state graph
        final_state = await self.graph.ainvoke(initial_state)
        
        # Return results
        result = {
            'response': final_state['final_response'],
            'retrieved_transactions_count': len(final_state['retrieved_transactions'])
        }
        
        logger.info(f"Query completed successfully")
        return result