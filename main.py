import json
import subprocess

from colorama import Fore, Style, init
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam

init(autoreset=True)

client = OpenAI()

messages: list[ChatCompletionMessageParam] = [
    {"role": "system", "content":
     "You are an expert coding agent that codes using the command line. You can use any Linux commands that make sense. If you execute a command the results will be given to you after the command completes."}
]

tools: list[ChatCompletionToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a shell command and return its output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute.",
                    }
                },
                "required": ["command"],
                "additionalProperties": False,
            },
        },
    }
]


def execute_bash(command):
    display_command = command.replace("\x00", "\\x00")
    print(f"{Fore.CYAN}[bash] command{Style.RESET_ALL}: {display_command}")

    if "\x00" in command:
        error_message = "Command contains an embedded null byte and was not executed."
        print(f"{Fore.RED}[bash] stderr{Style.RESET_ALL}: {error_message}")
        return json.dumps(
            {
                "command": display_command,
                "returncode": None,
                "stdout": "",
                "stderr": error_message,
            }
        )

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        print(f"{Fore.RED}[bash] stderr{Style.RESET_ALL}: Command timed out after 30 seconds.")
        return json.dumps(
            {
                "command": display_command,
                "returncode": None,
                "stdout": "",
                "stderr": "Command timed out after 30 seconds.",
            }
        )
    except (OSError, ValueError) as exc:
        error_message = str(exc)
        print(f"{Fore.RED}[bash] stderr{Style.RESET_ALL}: {error_message}")
        return json.dumps(
            {
                "command": display_command,
                "returncode": None,
                "stdout": "",
                "stderr": error_message,
            }
        )

    stdout_text = result.stdout or "<empty>"
    stderr_text = result.stderr or "<empty>"

    print(f"{Fore.GREEN}[bash] stdout{Style.RESET_ALL}: {stdout_text}")
    print(f"{Fore.RED}[bash] stderr{Style.RESET_ALL}: {stderr_text}")

    return json.dumps(
        {
            "command": display_command,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    )


def run_model_turn():
    while True:
        response = client.chat.completions.create(
            model="gpt-5.2",
            messages=messages,
            tools=tools,
        )

        message = response.choices[0].message

        if not message.tool_calls:
            assistant_reply = message.content or ""
            print(f"Assistant: {assistant_reply}")
            messages.append({"role": "assistant", "content": assistant_reply})
            return

        function_tool_calls = [
            tool_call for tool_call in message.tool_calls if tool_call.type == "function"
        ]

        messages.append(
            {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": tool_call.type,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                    for tool_call in function_tool_calls
                ],
            }
        )

        for tool_call in function_tool_calls:
            if tool_call.function.name != "bash":
                tool_output = json.dumps(
                    {"error": f"Unsupported tool: {tool_call.function.name}"}
                )
            else:
                arguments = json.loads(tool_call.function.arguments)
                tool_output = execute_bash(arguments["command"])

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_output,
                }
            )

print("Chat started. Type 'exit' or 'quit' to stop.")

while True:
    try:
        user_input = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye!")
        break

    if not user_input:
        continue

    if user_input.lower() in {"exit", "quit"}:
        print("Goodbye!")
        break

    messages.append({"role": "user", "content": user_input})
    run_model_turn()