import json
import subprocess
from pathlib import Path
from typing import Any

from colorama import Fore, Style, init
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam

init(autoreset=True)

client = OpenAI()
workspace_root = Path(__file__).resolve().parent

messages: list[ChatCompletionMessageParam] = [
    {"role": "system", "content":
     "You are an expert coding agent that codes using the command line. You can use Linux commands that make sense. You also have a think tool for visible reasoning and an apply_patch tool for updating files with diffs. If you execute a tool, the results will be given to you after the tool completes."}
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
    },
    {
        "type": "function",
        "function": {
            "name": "think",
            "description": "Record a thought for the user to see before continuing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "thought": {
                        "type": "string",
                        "description": "The thought to display to the user.",
                    }
                },
                "required": ["thought"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_patch",
            "description": "Apply a Linux patch-compatible diff to a file in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": ["update_file"],
                                "description": "The patch operation type.",
                            },
                            "diff": {
                                "type": "string",
                                "description": "A diff string that can be applied by the Linux patch command.",
                            },
                            "path": {
                                "type": "string",
                                "description": "The workspace-relative path to update.",
                            },
                        },
                        "required": ["type", "diff", "path"],
                        "additionalProperties": False,
                    }
                },
                "required": ["operation"],
                "additionalProperties": False,
            },
        },
    }
]


def print_tool_line(color: str, tool_name: str, label: str, value: str) -> None:
    print(f"{color}[{tool_name}] {label}{Style.RESET_ALL}: {value}")


def execute_think(thought: str) -> str:
    print_tool_line(Fore.MAGENTA, "think", "thought", thought)
    return json.dumps({"thought": thought})


def execute_apply_patch(operation: dict[str, Any]) -> str:
    operation_type = operation.get("type")
    path = operation.get("path")
    diff = operation.get("diff")

    print_tool_line(Fore.YELLOW, "apply_patch", "type", str(operation_type))
    print_tool_line(Fore.YELLOW, "apply_patch", "path", str(path))
    print_tool_line(Fore.BLUE, "apply_patch", "diff", str(diff))

    if operation_type != "update_file":
        error_message = f"Unsupported patch operation: {operation_type}"
        print_tool_line(Fore.RED, "apply_patch", "stderr", error_message)
        return json.dumps({"error": error_message})

    if not isinstance(path, str) or not path:
        error_message = "Patch operation path must be a non-empty string."
        print_tool_line(Fore.RED, "apply_patch", "stderr", error_message)
        return json.dumps({"error": error_message})

    if not isinstance(diff, str) or not diff:
        error_message = "Patch operation diff must be a non-empty string."
        print_tool_line(Fore.RED, "apply_patch", "stderr", error_message)
        return json.dumps({"error": error_message})

    try:
        target_path = (workspace_root / path).resolve()
        target_path.relative_to(workspace_root)
    except ValueError:
        error_message = "Patch path must stay within the workspace."
        print_tool_line(Fore.RED, "apply_patch", "stderr", error_message)
        return json.dumps({"error": error_message})

    if not target_path.is_file():
        error_message = f"Target file does not exist: {path}"
        print_tool_line(Fore.RED, "apply_patch", "stderr", error_message)
        return json.dumps({"error": error_message})

    patch_input = diff if diff.endswith("\n") else f"{diff}\n"

    try:
        result = subprocess.run(
            ["patch", "--forward", "--silent", str(target_path)],
            input=patch_input,
            capture_output=True,
            text=True,
            cwd=workspace_root,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        error_message = "Patch command timed out after 30 seconds."
        print_tool_line(Fore.RED, "apply_patch", "stderr", error_message)
        return json.dumps({"error": error_message})
    except (OSError, ValueError) as exc:
        error_message = str(exc)
        print_tool_line(Fore.RED, "apply_patch", "stderr", error_message)
        return json.dumps({"error": error_message})

    stdout_text = result.stdout or "<empty>"
    stderr_text = result.stderr or "<empty>"

    print_tool_line(Fore.GREEN, "apply_patch", "stdout", stdout_text)
    print_tool_line(Fore.RED, "apply_patch", "stderr", stderr_text)

    if result.returncode != 0:
        return json.dumps(
            {
                "error": "patch command failed",
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        )

    success_message = f"Updated {path}"
    return json.dumps(
        {
            "status": "ok",
            "path": path,
            "message": success_message,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    )


def execute_bash(command: str) -> str:
    display_command = command.replace("\x00", "\\x00")
    print_tool_line(Fore.CYAN, "bash", "command", display_command)

    if "\x00" in command:
        error_message = "Command contains an embedded null byte and was not executed."
        print_tool_line(Fore.RED, "bash", "stderr", error_message)
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
        print_tool_line(Fore.RED, "bash", "stderr", "Command timed out after 30 seconds.")
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
        print_tool_line(Fore.RED, "bash", "stderr", error_message)
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

    print_tool_line(Fore.GREEN, "bash", "stdout", stdout_text)
    print_tool_line(Fore.RED, "bash", "stderr", stderr_text)

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
                try:
                    arguments = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError as exc:
                    tool_output = json.dumps({"error": f"Invalid tool arguments: {exc}"})
                else:
                    if tool_call.function.name == "bash":
                        command = arguments.get("command")
                        if not isinstance(command, str):
                            tool_output = json.dumps(
                                {"error": "bash requires a string command argument."}
                            )
                        else:
                            tool_output = execute_bash(command)
                    elif tool_call.function.name == "think":
                        thought = arguments.get("thought")
                        if not isinstance(thought, str):
                            tool_output = json.dumps(
                                {"error": "think requires a string thought argument."}
                            )
                        else:
                            tool_output = execute_think(thought)
                    elif tool_call.function.name == "apply_patch":
                        operation = arguments.get("operation")
                        if not isinstance(operation, dict):
                            tool_output = json.dumps(
                                {"error": "apply_patch requires an operation object."}
                            )
                        else:
                            tool_output = execute_apply_patch(operation)
                    else:
                        tool_output = json.dumps(
                            {"error": f"Unsupported tool: {tool_call.function.name}"}
                        )

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