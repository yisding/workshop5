import json
import subprocess
from pathlib import Path
from typing import Any, Callable, Sequence, cast

from colorama import Back, Fore, Style, init
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam

init(autoreset=True)

client = OpenAI()
workspace_root = Path(__file__).resolve().parent


class ToolExecutionError(Exception):
    """Raised when a tool invocation is invalid or cannot be completed."""


def print_tool_line(color: str, tool_name: str, label: str, value: str) -> None:
    print(f"{color}[{tool_name}] {label}{Style.RESET_ALL}: {value}")


def print_agent_status(agent_name: str, status: str, color: str) -> None:
    banner = f"[{agent_name}] {status}"
    print(f"{Style.BRIGHT}{color}{banner}{Style.RESET_ALL}")


def print_tool_call(agent_name: str, tool_name: str) -> None:
    banner = f"[{agent_name}] TOOL CALL -> {tool_name}"
    print(f"{Style.BRIGHT}{Back.WHITE}{Fore.BLACK}{banner}{Style.RESET_ALL}")


def tool_bash(command: str) -> str:
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
            cwd=workspace_root,
        )
    except subprocess.TimeoutExpired:
        error_message = "Command timed out after 30 seconds."
        print_tool_line(Fore.RED, "bash", "stderr", error_message)
        return json.dumps(
            {
                "command": display_command,
                "returncode": None,
                "stdout": "",
                "stderr": error_message,
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


def tool_think(thought: str) -> str:
    print_tool_line(Fore.MAGENTA, "think", "thought", thought)
    return json.dumps({"thought": thought})


def tool_apply_patch(operation: dict[str, Any]) -> str:
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
        target_path = resolve_workspace_path(path)
    except ToolExecutionError as exc:
        error_message = str(exc)
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

    return json.dumps(
        {
            "status": "ok",
            "path": path,
            "message": f"Updated {path}",
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    )


def resolve_workspace_path(path: str) -> Path:
    try:
        resolved_path = (workspace_root / path).resolve()
        resolved_path.relative_to(workspace_root)
    except ValueError as exc:
        raise ToolExecutionError("Path must stay within the workspace.") from exc

    return resolved_path


class Agent:
    def __init__(
        self,
        *,
        name: str,
        model: str,
        system_prompt: str,
        tools: Sequence[ChatCompletionToolParam],
        messages: list[ChatCompletionMessageParam] | None = None,
    ) -> None:
        self.name = name
        self.model = model
        self.system_prompt = system_prompt
        self.tools = list(tools)
        self.messages = messages if messages is not None else []
        self._tool_handlers: dict[str, Callable[[dict[str, Any]], str]] = {}

    def register_tool_handler(
        self,
        tool_name: str,
        handler: Callable[[dict[str, Any]], str],
    ) -> None:
        self._tool_handlers[tool_name] = handler

    def ask(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})
        print_agent_status(self.name, "RUNNING", f"{Back.BLUE}{Fore.WHITE}")

        while True:
            system_message: ChatCompletionMessageParam = {
                "role": "system",
                "content": self.system_prompt,
            }
            messages_with_system = [system_message] + self.messages
            response = client.chat.completions.create(
                model=self.model,
                messages=messages_with_system,
                tools=self.tools,
            )

            message = response.choices[0].message
            if not message.tool_calls:
                assistant_reply = message.content or ""
                self.messages.append({"role": "assistant", "content": assistant_reply})
                print_agent_status(self.name, "COMPLETED", f"{Back.GREEN}{Fore.BLACK}")
                return assistant_reply

            function_tool_calls = [
                tool_call
                for tool_call in message.tool_calls
                if tool_call.type == "function"
            ]

            self.messages.append(
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
                tool_output = self._execute_tool_call(
                    tool_call.function.name,
                    tool_call.function.arguments,
                )
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_output,
                    }
                )

    def _execute_tool_call(self, tool_name: str, raw_arguments: str) -> str:
        print_tool_call(self.name, tool_name)
        handler = self._tool_handlers.get(tool_name)
        if handler is None:
            return json.dumps({"error": f"Unsupported tool: {tool_name}"})

        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            return json.dumps({"error": f"Invalid tool arguments: {exc}"})

        if not isinstance(arguments, dict):
            return json.dumps({"error": "Tool arguments must be a JSON object."})

        try:
            return handler(arguments)
        except ToolExecutionError as exc:
            return json.dumps({"error": str(exc)})
        except Exception as exc:  # defensive fallback
            return json.dumps({"error": f"Tool execution failed: {exc}"})


class CodeExplorationAgent(Agent):
    def __init__(self) -> None:
        tools = cast(
            list[ChatCompletionToolParam],
            [
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
                    "name": "summarize",
                    "description": "Summarize a file in the workspace using a separate LLM call.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Workspace-relative path to the file.",
                            },
                            "focus": {
                                "type": "string",
                                "description": "Optional focus area for the summary.",
                            },
                        },
                        "required": ["path"],
                        "additionalProperties": False,
                    },
                },
            },
            ],
        )

        super().__init__(
            name="code_explorer",
            model="gpt-5.2",
            system_prompt=(
                "You are a code exploration subagent. Perform read-only investigation, "
                "run shell commands as needed, and summarize files clearly. "
                "Do not attempt to modify files."
            ),
            tools=tools,
        )

        self.register_tool_handler("bash", self._handle_bash)
        self.register_tool_handler("think", self._handle_think)
        self.register_tool_handler("summarize", self._handle_summarize)

    @staticmethod
    def _handle_bash(arguments: dict[str, Any]) -> str:
        command = arguments.get("command")
        if not isinstance(command, str):
            raise ToolExecutionError("bash requires a string command argument.")
        return tool_bash(command)

    @staticmethod
    def _handle_think(arguments: dict[str, Any]) -> str:
        thought = arguments.get("thought")
        if not isinstance(thought, str):
            raise ToolExecutionError("think requires a string thought argument.")
        return tool_think(thought)

    @staticmethod
    def _handle_summarize(arguments: dict[str, Any]) -> str:
        path = arguments.get("path")
        focus = arguments.get("focus")

        if not isinstance(path, str) or not path:
            raise ToolExecutionError("summarize requires a non-empty string path.")
        if focus is not None and not isinstance(focus, str):
            raise ToolExecutionError("summarize focus must be a string when provided.")

        file_path = resolve_workspace_path(path)
        if not file_path.is_file():
            raise ToolExecutionError(f"File does not exist: {path}")

        try:
            contents = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ToolExecutionError(
                f"Unable to decode {path} as UTF-8 text: {exc}"
            ) from exc
        except OSError as exc:
            raise ToolExecutionError(f"Unable to read file: {exc}") from exc

        max_chars = 20000
        truncated = len(contents) > max_chars
        contents_for_summary = contents[:max_chars]

        summarize_messages: list[ChatCompletionMessageParam] = [
            {
                "role": "system",
                "content": (
                    "You summarize source files. Provide a concise technical summary with "
                    "purpose, key symbols, and notable behaviors. Mention important risks "
                    "or oddities if present."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"File: {path}\n"
                    f"Focus: {focus or 'General architecture and behavior'}\n"
                    "If content appears truncated, state that in the summary.\n\n"
                    f"```\n{contents_for_summary}\n```"
                ),
            },
        ]

        summary_response = client.chat.completions.create(
            model="gpt-5.2",
            messages=summarize_messages,
        )

        summary_text = summary_response.choices[0].message.content or ""
        print_tool_line(Fore.MAGENTA, "summarize", "path", path)

        return json.dumps(
            {
                "path": path,
                "focus": focus,
                "truncated": truncated,
                "summary": summary_text,
            }
        )


class PlanningAgent(Agent):
    def __init__(
        self,
        explorer: "CodeExplorationAgent",
        messages: list[ChatCompletionMessageParam] | None = None,
    ) -> None:
        self.explorer = explorer

        tools = cast(
            list[ChatCompletionToolParam],
            [
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
                    "name": "explore",
                    "description": "Delegate read-only code exploration to a dedicated subagent.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "What the subagent should investigate.",
                            }
                        },
                        "required": ["prompt"],
                        "additionalProperties": False,
                    },
                },
            },
            ],
        )

        super().__init__(
            name="planning_agent",
            model="gpt-5.2",
            system_prompt=(
                "You are a planning agent. Generate detailed bullet-point plans for coding tasks. "
                "Use bash and explore tools to investigate the codebase, then produce a comprehensive, "
                "step-by-step plan without modifying any files. Focus on clarity and actionability."
            ),
            tools=tools,
            messages=messages,
        )

        self.register_tool_handler("bash", self._handle_bash)
        self.register_tool_handler("think", self._handle_think)
        self.register_tool_handler("explore", self._handle_explore)

    @staticmethod
    def _handle_bash(arguments: dict[str, Any]) -> str:
        command = arguments.get("command")
        if not isinstance(command, str):
            raise ToolExecutionError("bash requires a string command argument.")
        return tool_bash(command)

    @staticmethod
    def _handle_think(arguments: dict[str, Any]) -> str:
        thought = arguments.get("thought")
        if not isinstance(thought, str):
            raise ToolExecutionError("think requires a string thought argument.")
        return tool_think(thought)

    def _handle_explore(self, arguments: dict[str, Any]) -> str:
        prompt = arguments.get("prompt")
        if not isinstance(prompt, str):
            raise ToolExecutionError("explore requires a string prompt argument.")

        print_tool_line(Fore.CYAN, "explore", "prompt", prompt)
        exploration_reply = self.explorer.ask(prompt)
        print_tool_line(Fore.GREEN, "explore", "result", exploration_reply)

        return json.dumps({"prompt": prompt, "result": exploration_reply})


class CodingAgent(Agent):
    def __init__(
        self,
        explorer: CodeExplorationAgent,
        messages: list[ChatCompletionMessageParam] | None = None,
    ) -> None:
        self.explorer = explorer

        tools = cast(
            list[ChatCompletionToolParam],
            [
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
            },
            {
                "type": "function",
                "function": {
                    "name": "explore",
                    "description": "Delegate read-only code exploration to a dedicated subagent.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "What the subagent should investigate.",
                            }
                        },
                        "required": ["prompt"],
                        "additionalProperties": False,
                    },
                },
            },
            ],
        )

        super().__init__(
            name="coding_agent",
            model="gpt-5.2",
            system_prompt=(
                "You are an expert coding agent that codes using the command line. "
                "Use Linux commands that make sense. You have bash, think, apply_patch, "
                "and an explore tool that calls a read-only code exploration subagent. "
                "If you execute a tool, the results will be provided after it completes."
                "\n"
                "\n"
                "When the user asks for information about a large codebase, call explore with"
                " different prompts to investigate the codebase in a structured way."
                " For example, you might start with 'explore: summarize the architecture of the codebase', then "
                " follow up with 'explore: summarize the main loop in src/main.py' or 'explore: what does the tests/ directory contain?'"
            ),
            tools=tools,
            messages=messages,
        )

        self.register_tool_handler("bash", self._handle_bash)
        self.register_tool_handler("think", self._handle_think)
        self.register_tool_handler("apply_patch", self._handle_apply_patch)
        self.register_tool_handler("explore", self._handle_explore)

    @staticmethod
    def _handle_bash(arguments: dict[str, Any]) -> str:
        command = arguments.get("command")
        if not isinstance(command, str):
            raise ToolExecutionError("bash requires a string command argument.")
        return tool_bash(command)

    @staticmethod
    def _handle_think(arguments: dict[str, Any]) -> str:
        thought = arguments.get("thought")
        if not isinstance(thought, str):
            raise ToolExecutionError("think requires a string thought argument.")
        return tool_think(thought)

    @staticmethod
    def _handle_apply_patch(arguments: dict[str, Any]) -> str:
        operation = arguments.get("operation")
        if not isinstance(operation, dict):
            raise ToolExecutionError("apply_patch requires an operation object.")
        return tool_apply_patch(operation)

    def _handle_explore(self, arguments: dict[str, Any]) -> str:
        prompt = arguments.get("prompt")
        if not isinstance(prompt, str):
            raise ToolExecutionError("explore requires a string prompt argument.")

        print_tool_line(Fore.CYAN, "explore", "prompt", prompt)
        exploration_reply = self.explorer.ask(prompt)
        print_tool_line(Fore.GREEN, "explore", "result", exploration_reply)

        return json.dumps({"prompt": prompt, "result": exploration_reply})


def main() -> None:
    explorer = CodeExplorationAgent()
    shared_messages: list[ChatCompletionMessageParam] = []
    coding_agent = CodingAgent(explorer=explorer, messages=shared_messages)
    planning_agent = PlanningAgent(explorer=explorer, messages=shared_messages)

    current_agent = coding_agent
    print("Chat started. Type 'exit' or 'quit' to stop.")
    print("Use '/plan' to switch to the planning agent or '/code' to switch to the coding agent.")

    while True:
        try:
            agent_label = f"[{current_agent.name}]" if current_agent else "[chat]"
            user_input = input(f"You {agent_label}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if user_input.lower() == "/plan":
            current_agent = planning_agent
            print(f"\n{Style.BRIGHT}{Back.MAGENTA}{Fore.WHITE}Switched to planning agent{Style.RESET_ALL}\n")
            continue

        if user_input.lower() == "/code":
            current_agent = coding_agent
            print(f"\n{Style.BRIGHT}{Back.CYAN}{Fore.BLACK}Switched to coding agent{Style.RESET_ALL}\n")
            continue

        assistant_reply = current_agent.ask(user_input)
        print(f"Assistant: {assistant_reply}")


if __name__ == "__main__":
    main()
