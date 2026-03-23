from openai import OpenAI
import json

client = OpenAI()

# System prompts for each agent
DOG_AGENT_SYSTEM_PROMPT = "You should answer questions about dogs. If the question isn't about dogs, call the 'transferToCatAgent' function."
CAT_AGENT_SYSTEM_PROMPT = "You should answer questions about cats. If the question isn't about cats, call the 'transferToDogAgent' function."

# Tools for dog agent
dog_agent_tools = [
    {
        "type": "function",
        "function": {
            "name": "transferToCatAgent",
            "description": "Transfer the conversation to the cat agent when the question is about cats",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False
            }
        }
    }
]

# Tools for cat agent
cat_agent_tools = [
    {
        "type": "function",
        "function": {
            "name": "transferToDogAgent",
            "description": "Transfer the conversation to the dog agent when the question is about dogs",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False
            }
        }
    }
]

def clean_messages_for_transfer(messages):
    """Remove all function calls and tool results from message history."""
    cleaned = []
    for msg in messages:
        # Skip tool messages
        if msg.get("role") == "tool":
            continue
        # Skip assistant messages with tool calls
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            continue
        # Handle dict messages
        if isinstance(msg, dict):
            if msg.get("role") == "tool":
                continue
            if msg.get("tool_calls"):
                continue
            # Skip system messages (we'll add the new agent's system prompt)
            if msg.get("role") == "system":
                continue
            cleaned.append(msg)
    return cleaned

def call_agent(agent_name, messages, tools, system_prompt):
    """Call an agent and return the response."""
    # Prepare messages with system prompt
    full_messages = [{"role": "system", "content": system_prompt}] + messages
    
    print(f"\n{'='*50}")
    print(f"[LLM REQUEST] Calling {agent_name}")
    print(f"Messages: {json.dumps(full_messages, indent=2, default=str)}")
    print(f"Tools: {[t['function']['name'] for t in tools]}")
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=full_messages,
        tools=tools
    )
    
    assistant_message = response.choices[0].message
    print(f"\n[LLM RESPONSE] From {agent_name}")
    print(f"Content: {assistant_message.content}")
    print(f"Tool calls: {assistant_message.tool_calls}")
    
    return assistant_message

def run_conversation(user_message):
    """Run the multi-agent conversation until we get a final answer."""
    # Start with dog agent
    current_agent = "dog"
    messages = [{"role": "user", "content": user_message}]
    
    while True:
        if current_agent == "dog":
            system_prompt = DOG_AGENT_SYSTEM_PROMPT
            tools = dog_agent_tools
            agent_name = "Dog Agent"
        else:
            system_prompt = CAT_AGENT_SYSTEM_PROMPT
            tools = cat_agent_tools
            agent_name = "Cat Agent"
        
        assistant_message = call_agent(agent_name, messages, tools, system_prompt)
        
        # Check if there are tool calls
        if not assistant_message.tool_calls:
            # No tool calls, we have a final answer
            print(f"\n{'='*50}")
            print(f"[FINAL ANSWER] From {agent_name}")
            print(assistant_message.content)
            return assistant_message.content
        
        # Process tool calls
        for tool_call in assistant_message.tool_calls:
            function_name = tool_call.function.name
            function_args = tool_call.function.arguments
            
            print(f"\n[FUNCTION CALL] {function_name}")
            print(f"Arguments: {function_args}")
            
            if function_name == "transferToCatAgent":
                print("[TRANSFER] Switching to Cat Agent")
                # Clean messages and switch agent
                messages = clean_messages_for_transfer(messages)
                current_agent = "cat"
            elif function_name == "transferToDogAgent":
                print("[TRANSFER] Switching to Dog Agent")
                # Clean messages and switch agent
                messages = clean_messages_for_transfer(messages)
                current_agent = "dog"

# Test the conversation
if __name__ == "__main__":
    user_input = input("Ask a question about dogs or cats: ")
    run_conversation(user_input)

