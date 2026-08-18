"""LLM engine using Ollama with tool support."""

import logging
from typing import List, Optional

import ollama

from .web_search import web_search

logger = logging.getLogger(__name__)

# Define available tools
TOOLS = {
    "web_search": {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information on any topic. Use this when users ask about real-time information like news, weather, sports scores, stock prices, or any current events.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up on the web"
                    }
                },
                "required": ["query"]
            }
        }
    }
}

class LLMEngine:
    def __init__(self, model_name: str = "llama3.1:8b",
                 host: str = "http://localhost:11434"):
        """
        Initialize the LLM engine.

        Args:
            model_name: Name of the Ollama model to use
            host: Ollama server host
        """
        self.model_name = model_name
        self.host = host
        self.client = ollama.Client(host=host)
        # Test connection
        try:
            self.client.list()
            logger.info(f"Connected to Ollama at {host}")
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            raise

    def generate(self, prompt: str, system: Optional[str] = None,
                 stream: bool = False) -> str:
        """
        Generate a response from the LLM.

        Args:
            prompt: The user's input prompt
            system: System message to set behavior
            stream: Whether to stream the response

        Returns:
            Generated text response
        """
        try:
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                system=system,
                stream=stream
            )
            if stream:
                # Handle streaming response
                full_response = ""
                for chunk in response:
                    full_response += chunk['response']
                return full_response
            else:
                return response['response']
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            return f"Error: {str(e)}"

    def chat(self, messages: List[dict], stream: bool = False) -> str:
        """
        Chat with the LLM using a message history, with tool support.

        Args:
            messages: List of message dicts with 'role' and 'content'
            stream: Whether to stream the response

        Returns:
            Generated text response
        """
        try:
            # Prepare tools for the model
            tools = list(TOOLS.values())

            # Helper to call chat with tool_choice if supported
            def _ollama_chat(**kwargs):
                try:
                    # Try with tool_choice="required" to encourage tool use
                    return self.client.chat(
                        model=self.model_name,
                        messages=kwargs.get('messages'),
                        tools=kwargs.get('tools', []),
                        tool_choice="required",
                        stream=kwargs.get('stream', False)
                    )
                except TypeError:
                    # Fallback if tool_choice not supported
                    return self.client.chat(
                        model=self.model_name,
                        messages=kwargs.get('messages'),
                        tools=kwargs.get('tools', []),
                        stream=kwargs.get('stream', False)
                    )

            response = _ollama_chat(
                messages=messages,
                tools=tools,
                stream=stream
            )

            if stream:
                full_response = ""
                for chunk in response:
                    if 'message' in chunk:
                        if 'content' in chunk['message']:
                            full_response += chunk['message']['content']
                        # Handle tool calls in streaming (if supported)
                        # Note: Ollama's streaming tool call support may vary
                return full_response
            else:
                # Handle tool calls in non-streaming response
                message = response['message']

                # Check if the model wants to use tools
                if 'tool_calls' in message and message['tool_calls']:
                    logger.info(f"LLM requested tool calls: {message['tool_calls']}")

                    # Add the assistant's message to conversation
                    messages.append(message)

                    # Execute each tool call
                    for tool_call in message['tool_calls']:
                        function_name = tool_call['function']['name']
                        function_args = tool_call['function']['arguments']

                        logger.info(f"Executing tool: {function_name} with args: {function_args}")

                        # Execute the tool
                        if function_name in TOOLS:
                            if function_name == "web_search":
                                query = function_args.get('query', '')
                                # Note: In the LLM engine context, we don't have direct access to persona config
                                # The search provider will be determined by environment variable or default to tavily
                                result = web_search(query)

                                # Add tool result to messages
                                messages.append({
                                    "role": "tool",
                                    "content": result,
                                    "tool_call_id": tool_call.get('id', ''),
                                    "name": function_name
                                })
                            else:
                                # Unknown tool
                                messages.append({
                                    "role": "tool",
                                    "content": f"Error: Unknown tool {function_name}",
                                    "tool_call_id": tool_call.get('id', ''),
                                    "name": function_name
                                })
                        else:
                            # Tool not found
                            messages.append({
                                "role": "tool",
                                "content": f"Error: Tool {function_name} not available",
                                "tool_call_id": tool_call.get('id', ''),
                                "name": function_name
                            })

                    # Get final response from LLM after tool execution
                    final_response = _ollama_chat(
                        model=self.model_name,
                        messages=messages,
                        tools=tools,
                        stream=False  # Don't stream the final response for simplicity
                    )
                    return final_response['message']['content']
                else:
                    # No tool calls, return regular response
                    return message['content']

        except Exception as e:
            logger.error(f"Error in LLM chat: {e}")
            return f"Error: {str(e)}"

# Convenience function for simple usage
def generate_response(prompt: str, model_name: str = "llama3.1:8b",
                     system: Optional[str] = None) -> str:
    """
    Generate a response from the LLM using a temporary engine.

    Args:
        prompt: The user's input prompt
        model_name: Ollama model name
        system: System message

    Returns:
        Generated text response
    """
    engine = LLMEngine(model_name=model_name)
    return engine.generate(prompt, system=system)