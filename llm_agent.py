from typing import TypedDict, List, Optional, Annotated, Literal
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
# from tools.plot_tool import create_financial_plot
from tools.qdrant_tool import retrieve_transactions
# from tools.metrics_tool import calculate_financial_metrics
from pydantic import SecretStr
import matplotlib.pyplot as plt
from config import OPENAI_KEY, OPENAI_MODEL_NAME, GEMINI_KEY, GEMINI_MODEL_NAME, get_logger

logger = get_logger(__name__)

# Define the state structure
class AgentState(TypedDict):
    messages: Annotated[List, "The conversation history"]
    user_query: str
    user_id: str  # Add user_id to state
    tool_calls: Optional[List]
    next_action: Optional[str]
    retrieved_transactions: Optional[List[dict]]
    plot_data: Optional[str]  # Base64 encoded plot
    search_results: Optional[str]
    final_response: Optional[str]


class LLMAgent:
    def __init__(self):
        logger.info("Initializing LLM Agent")
        # Initialize LLM with tool calling capability
        self.llm = ChatGoogleGenerativeAI(
            api_key=GEMINI_KEY, 
            model=GEMINI_MODEL_NAME, 
            disable_streaming=False, 
            temperature=0.5,
        ).bind_tools([
            retrieve_transactions,
            # create_financial_plot,
            # calculate_financial_metrics
        ])
        logger.info(f"LLM initialized with model: {GEMINI_MODEL_NAME}")
        
        # Initialize tool executor
        self.tools = [
            retrieve_transactions, 
            # create_financial_plot, 
            # calculate_financial_metrics
        ]
        logger.info(f"Tools initialized: {[tool.name for tool in self.tools]}")
        
        # Build the graph
        self.graph = self._build_graph()
        logger.info("Agent workflow graph built")
    
    def _build_graph(self):
        """Build the LangGraph workflow with tool routing"""
        logger.debug("Building agent workflow graph")
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("tools", self._tool_node)
        workflow.add_node("generate_response", self._generate_response_node)
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "continue": "tools",
                "end": "generate_response"
            }
        )
        workflow.add_edge("tools", "agent")
        workflow.add_edge("generate_response", END)
        
        # Set entry point
        workflow.set_entry_point("agent")
        
        logger.debug("Workflow graph construction completed")
        return workflow.compile()
    
    def _agent_node(self, state: AgentState) -> AgentState:
        """Main agent decision-making node"""
        logger.info("Agent node: Processing user query")
        logger.debug(f"User query: {state['user_query']}")
        logger.debug(f"User ID: {state['user_id']}")
        
        system_prompt = f"""You are a financial assistant with access to multiple tools:

1. retrieve_transactions: Get transaction data from the database for user {state['user_id']}
2. create_financial_plot: Create visualizations of financial data  
3. search_financial_news: Search for recent financial news
4. calculate_financial_metrics: Calculate financial metrics and insights

For user queries:
- If they want transaction data, use retrieve_transactions first (always include user_id: {state['user_id']})
- If they want visualizations, use create_financial_plot after getting data
- If they ask about market news or external financial info, use search_financial_news
- If they want insights/metrics, use calculate_financial_metrics
- You can use multiple tools in sequence to fully answer their question

Be intelligent about tool selection and provide comprehensive responses."""
        
        messages: list[BaseMessage] = [SystemMessage(content=system_prompt)]
        
        # Add conversation history
        if state.get('messages'):
            messages.extend(state['messages'][-6:])  # Last 6 messages for context
            logger.debug(f"Added {len(state['messages'][-6:])} messages from history")
        
        messages.append(HumanMessage(content=state['user_query']))
        
        logger.info("Calling LLM for decision")
        response = self.llm.invoke(messages)
        logger.debug(f"LLM response: {response}")
        
        # Update state with the agent's response
        state['messages'] = state.get('messages', []) + [response]
        
        return state
    
    def _tool_node(self, state: AgentState) -> AgentState:
        """Execute tools based on agent's decisions"""
        logger.info("Tool node: Executing tools")
        
        last_message = state['messages'][-1]
        
        # Execute tool calls
        tool_outputs = []
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            logger.info(f"Found {len(last_message.tool_calls)} tool calls to execute")
            
            # Get already executed tool IDs
            executed_tool_ids = {msg.tool_call_id for msg in state['messages'] if isinstance(msg, ToolMessage)}
            
            for tool_call in last_message.tool_calls:
                # Skip if we've already executed this tool call
                if tool_call['id'] in executed_tool_ids:
                    logger.info(f"Skipping already executed tool call: {tool_call['id']}")
                    continue
                    
                # Find the tool by name
                tool_name = tool_call['name']
                logger.info(f"Executing tool: {tool_name}")
                logger.debug(f"Tool arguments: {tool_call['args']}")
                
                tool = next((t for t in self.tools if t.name == tool_name), None)
                
                if tool:
                    try:
                        # Execute the tool with the provided arguments
                        tool_output = tool.invoke(tool_call['args'])
                        logger.info(f"Tool {tool_name} executed successfully")
                        logger.debug(f"Tool output: {tool_output}")
                        
                        tool_outputs.append(ToolMessage(
                            content=str(tool_output),
                            tool_call_id=tool_call['id']
                        ))
                    except Exception as e:
                        logger.error(f"Error executing tool {tool_name}: {str(e)}", exc_info=True)
                        tool_outputs.append(ToolMessage(
                            content=f"Error executing tool: {str(e)}",
                            tool_call_id=tool_call['id']
                        ))
                else:
                    logger.error(f"Tool {tool_name} not found")
                    tool_outputs.append(ToolMessage(
                        content=f"Tool {tool_name} not found",
                        tool_call_id=tool_call['id']
                    ))
        else:
            logger.info("No tool calls found in last message")
        
        # Add tool outputs to messages
        state['messages'].extend(tool_outputs)
        
        return state
    
    def _should_continue(self, state: AgentState) -> Literal["continue", "end"]:
        """Determine whether to continue with tools or generate final response"""
        logger.info(f"Checking if should continue with tools: {state}")
        
        last_message = state['messages'][-1]
        
        # If we already have a final response, end the workflow
        if state.get('final_response'):
            logger.info("Found existing final response, ending workflow")
            return "end"
            
        # If the last message is a tool message, check if we need more tool calls
        if isinstance(last_message, ToolMessage):
            # Look at the previous message to see if it had tool calls
            if len(state['messages']) > 1:
                prev_message = state['messages'][-2]
                if hasattr(prev_message, 'tool_calls') and prev_message.tool_calls:
                    # Check if we've executed all the tool calls from the previous message
                    executed_tool_ids = {msg.tool_call_id for msg in state['messages'] if isinstance(msg, ToolMessage)}
                    pending_tool_ids = {tool_call['id'] for tool_call in prev_message.tool_calls}
                    
                    if not pending_tool_ids.issubset(executed_tool_ids):
                        logger.info("Some tool calls still pending, continuing to tools")
                        return "continue"
            
            logger.info("All tool calls completed, moving to final response generation")
            return "end"
            
        # If the last message has tool calls, continue to tools
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            # Check if we've already executed all these tool calls
            executed_tool_ids = {msg.tool_call_id for msg in state['messages'] if isinstance(msg, ToolMessage)}
            pending_tool_ids = {tool_call['id'] for tool_call in last_message.tool_calls}
            
            if not pending_tool_ids.issubset(executed_tool_ids):
                logger.info("Found new tool calls, continuing to tools")
                return "continue"
            else:
                logger.info("All tool calls already executed, moving to final response generation")
                return "end"
            
        # If the last message has content, we can end
        if hasattr(last_message, 'content') and last_message.content:
            logger.info("Found content in last message, ending workflow")
            return "end"
            
        logger.info("No clear end condition, moving to final response generation")
        return "end"
    
    def _generate_response_node(self, state: AgentState) -> AgentState:
        """Generate final human-readable response"""
        logger.info("Generating final response")
        
        # Get the last message that contains the transaction data
        last_message = state['messages'][-1]
        if isinstance(last_message, ToolMessage):
            logger.info("Processing tool response to generate final response")
            system_prompt = """Based on the tool response, provide a comprehensive, 
            helpful response to the user. Format the transaction data in a clear, readable way.
            If there are multiple transactions, group them by category or date if appropriate.
            Be conversational and helpful."""
            
            messages = [SystemMessage(content=system_prompt)]
            messages.extend(state['messages'])
            
            logger.info("Calling LLM to process tool response")
            response = self.llm.invoke(messages)
            logger.debug(f"Final response: {response.text()}")
            
            # Store the response
            state['final_response'] = response.text()
            logger.info(f"Stored final response: {state['final_response'][:100]}...")
            return state
        
        # If no tool message, try to generate a new response
        system_prompt = """Based on the conversation and tool results, provide a comprehensive, 
        helpful response to the user. If plots were created, mention that visualizations are available.
        If financial data was retrieved, provide insights and summaries. Be conversational and helpful."""
        
        messages = [SystemMessage(content=system_prompt)]
        messages.extend(state['messages'])
        
        logger.info("Calling LLM for final response")
        response = self.llm.invoke(messages)
        logger.debug(f"Final response: {response.text()}")
        
        # Store the response
        state['final_response'] = response.text()
        logger.info(f"Stored final response: {state['final_response'][:100]}...")  # Log first 100 chars
        
        return state
    
    async def query(self, user_query: str, user_id: str, chat_history: List = []) -> dict:
        """Process a user query and return comprehensive results
        
        Args:
            user_query: The user's query text
            user_id: The ID of the user making the query
            chat_history: Optional list of previous messages
        """
        logger.info(f"Processing new query for user {user_id}")
        logger.debug(f"Query: {user_query}")
        logger.debug(f"Chat history length: {len(chat_history)}")
        
        initial_state = {
            'messages': chat_history or [],
            'user_query': user_query,
            'user_id': user_id,  # Add user_id to initial state
            'tool_calls': None,
            'next_action': None,
            'retrieved_transactions': None,
            'plot_data': None,
            'search_results': None,
            'final_response': None
        }
        
        # Run the workflow
        logger.info("Starting workflow execution")
        final_state = await self.graph.ainvoke(initial_state)
        logger.info("Workflow execution completed")
        
        # Extract results
        result = final_state.get('final_response', 'No response generated')
        logger.debug(f"Final response: {result}")
        
        # Check if any plots were created in the conversation
        for message in final_state.get('messages', []):
            if isinstance(message, ToolMessage) and 'base64' in str(message.content):
                logger.info("Found plot data in response")
                result = {'response': result, 'plot_data': str(message.content)}
                break
        
        # If no plot data was found, wrap the response in a dict
        if not isinstance(result, dict):
            result = {'response': result}
            
        logger.info(f"Returning final result: {str(result)[:100]}...")  # Log first 100 chars
        return result
