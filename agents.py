"""
Autonomous Coding Agent Module.

This module provides a flexible Agent class for creating autonomous coding agents that
can interact with the OpenAI Chat API to perform tasks using tool-calling capabilities.

Key components:
- Agent: The main class for creating autonomous coding agents
- LLMClient: A thin wrapper around the OpenAI Chat Completions API
- ChatMessage: A lightweight wrapper for chat messages
- ConversationMemory: Manages conversation history with optional summarization

Usage:
    from agents import Agent, LLMClient
    
    # Create an LLM client
    model = LLMClient(model_id="gpt-4.1")
    
    # Create an agent with tools
    agent = Agent(
        tools=[WriteFile(), ReadFile(), ...],
        model=model,
        name="my_agent",
        max_steps=20
    )
    
    # Run the agent on a task
    result = agent.run("Create a simple Python web server")

Author: Chris Weber, crweber@gmail.com
"""

from __future__ import annotations

###############################################################################
# Imports
###############################################################################

# stdlib ---------------------------------------------------------------------
import argparse
import datetime as _dt
import json
import logging
import os
import re
import textwrap
import traceback
import uuid
from dataclasses import dataclass, field
from typing import (Any, Callable, Dict, Iterable, List, Mapping, MutableMapping,
                    MutableSequence, Optional, Sequence, Union)

# local ----------------------------------------------------------------------
from agent_tools import (
    Tool, WriteFile, ReadFile, EditFile, Delete, RunPython,
    RunBash, ViewImage, ListFiles, MakePlan, FinalAnswer, GetUserInput,
    truncate, authorized_types, _RE_TRAILING_COMMA
)

# 3rd-party ------------------------------------------------------------------
try:
    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.text import Text

    _console: Console | None = Console()
except ImportError:  # pragma: no cover - Rich not installed

    class _DummyConsole:  # noqa: D401
        """Fallback console that prints plain text."""

        def print(self, *args: Any, **_: Any) -> None:  # noqa: D401
            print(*args)

    _console = _DummyConsole()

try:
    from openai import OpenAI  # type: ignore
except ImportError as exc:  # pragma: no cover - optional dependency
    raise SystemExit("`openai` package missing - install it to run the agent") from exc

###############################################################################
# Helpers & constants
###############################################################################

LOGGER = logging.getLogger("agent")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s ▸ %(message)s")

# Global configuration flags
CONFIRM_EDITS = False  # Set to True to enable interactive edit approval workflow

###############################################################################
# Chat message & memory classes
###############################################################################

# near the top of agent.py (after imports)

def _normalise_tool_calls(raw):
    """
    Return a list of JSON-serialisable dicts regardless of whether *raw*
    contains ChatCompletionMessageToolCall objects or ordinary dicts.
    """
    if not raw:
        return None
    result = []
    for tc in raw:
        if isinstance(tc, dict):          # already OK
            result.append(tc)
        else:                             # OpenAI pydantic object
            result.append({
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            })
    return result

@dataclass
class ChatMessage:
    """Lightweight wrapper around a single chat message."""

    role: str
    content: str | None = None
    # tool call fields --------------------------------------------------------
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: Any | None = None  # populated for assistant messages
    usage: Any | None = None  # stores token usage information

    # ---------------------------------------------------------------------
    # Factory helpers
    # ---------------------------------------------------------------------
    @classmethod
    def system(cls, content: str) -> "ChatMessage":  # noqa: D401
        return cls("system", content)

    @classmethod
    def user(cls, content: str) -> "ChatMessage":  # noqa: D401
        return cls("user", content)

    @classmethod
    def assistant(cls, content: str | None, *, tool_calls: Any | None = None):
        return cls("assistant", content, tool_calls=_normalise_tool_calls(tool_calls))

    @classmethod
    def tool(cls, *, name: str, tool_call_id: str, result: str) -> "ChatMessage":  # noqa: D401
        return cls("tool", result, name=name, tool_call_id=tool_call_id)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------
    def to_openai(self) -> Dict[str, Any]:  # noqa: D401
        msg: Dict[str, Any] = {"role": self.role}
        if self.name:
            msg["name"] = self.name
        if self.role == "tool":
            msg["tool_call_id"] = self.tool_call_id
            msg["content"] = self.content or ""
        else:
            msg["content"] = self.content or ""
            if self.tool_calls is not None:
                msg["tool_calls"] = self.tool_calls
        return msg

    # ------------------------------------------------------------------
    # Rich pretty-print helpers - optional
    # ------------------------------------------------------------------
    def _pretty(self) -> Panel:  # pragma: no cover - console only
        assert _console is not None  # appease mypy
        
        # Find agent name
        agent_name = "agent"
        import inspect
        frame = inspect.currentframe()
        while frame:
            if 'self' in frame.f_locals and hasattr(frame.f_locals['self'], 'name'):
                if isinstance(frame.f_locals['self'].name, str):
                    agent_name = frame.f_locals['self'].name
                    break
            frame = frame.f_back
            
        # Format based on message type
        if self.role == "assistant" and self.tool_calls:
            # Process tool calls for display
            tool_parts = []
            
            for tc in self.tool_calls:
                func_name = tc["function"]["name"]
                args_raw = tc["function"].get("arguments", "{}")
                
                # Format arguments (handle both string and dict forms)
                try:
                    args_obj = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                    
                    # Properly handle string values with escaped newlines and long text
                    if isinstance(args_obj, dict):
                        for k, v in args_obj.items():
                            if isinstance(v, str):
                                # Replace literal \n with actual newlines
                                if '\\n' in v:
                                    v = v.replace('\\n', '\n')
                                
                                # Wrap long lines
                                if len(v) > 80:
                                    # For text with newlines, apply wrapping to each line
                                    if '\n' in v:
                                        wrapped_lines = []
                                        for line in v.split('\n'):
                                            if len(line) > 80:
                                                wrapped_lines.append(textwrap.fill(line, width=80))
                                            else:
                                                wrapped_lines.append(line)
                                        v = '\n'.join(wrapped_lines)
                                    else:
                                        v = textwrap.fill(v, width=80)
                                
                                args_obj[k] = v
                    
                    args_str = json.dumps(args_obj, indent=2, ensure_ascii=False)
                    # Unescape newlines for display
                    args_str = args_str.replace('\\n', '\n')
                except Exception:
                    args_str = str(args_raw)
                    
                tool_parts.append(f"Tool: {func_name}\nArguments:\n{args_str}")
                
            # Create tool syntax display with word wrapping enabled
            tool_syntax = Syntax("\n\n".join(tool_parts), "yaml", theme="ansi_dark", word_wrap=True)
            
            return Panel(
                Group(
                    Text(f"ASSISTANT [{agent_name}]", style="bold cyan"),
                    Text(self.content or "", style="white"),
                    Text("\nTOOL CALLS:", style="bold yellow"),
                    tool_syntax
                ),
                title="assistant",
                border_style="cyan",
                width=100,
                expand=True
            )
        if self.role == "tool":
            raw = self.content or ""
            # Hide giant data-URI strings from view_image (or any tool returning one)
            if self.name == "view_image" or raw.lstrip().startswith("data:image"):
                shown = f"<image data-URI - {len(raw):,} characters>"
            else:
                shown = raw
            return Panel(
                Group(
                    Text(f"TOOL RESPONSE - {self.name}", style="bold blue"),
                    Text(shown, style="white")
                ),
                title="tool",
                border_style="blue"
            )
        return Panel(Group(Text(self.role.upper(), style="bold magenta"), Text(self.content or "", style="white")),
                     border_style="magenta", title=self.role)


class ConversationMemory(MutableSequence[ChatMessage]):
    """Conversation history with optional automatic summarisation."""

    def __init__(self) -> None:
        self._messages: list[ChatMessage] = []

    # MutableSequence interface ---------------------------------------------
    def __getitem__(self, idx: int) -> ChatMessage:  # noqa: D401
        return self._messages[idx]

    def __setitem__(self, idx: int, value: ChatMessage) -> None:  # noqa: D401
        self._messages[idx] = value

    def __delitem__(self, idx: int) -> None:  # noqa: D401
        del self._messages[idx]

    def __len__(self) -> int:  # noqa: D401
        return len(self._messages)

    def insert(self, idx: int, value: ChatMessage) -> None:  # noqa: D401
        self._messages.insert(idx, value)

    # Convenience ------------------------------------------------------------
    def to_openai(self) -> List[Dict[str, Any]]:  # noqa: D401
        return [m.to_openai() for m in self._messages]
        
    # Memory summarization -------------------------------------------------
    def summarize(self, model: Any) -> None:
        """
        Summarize the middle of the conversation when it grows too large.
        
        Keeps:
        - first 2 messages (system + first user)
        - optional plan at index 2 
        - last 4 messages for local context
        
        Then inserts an assistant-authored summary in the middle.
        """
        if len(self._messages) <= 6:  # Not enough to summarize
            return
            
        # Keep head (system + first user + optional plan)
        head = list(self._messages[:2])
        if len(self._messages) > 2 and self._messages[2].role == "assistant":
            head.append(self._messages[2])  # keep initial plan
            
        # Keep tail (last 4 messages)
        tail_len = 4
        tail = list(self._messages[-tail_len:])
        
        # Skip leading tool responses in tail
        while tail and tail[0].role == "tool":
            tail.pop(0)
            if len(tail) < 2 and len(self._messages) > len(head) + 2:
                # Grab more from the original to ensure we have at least a couple messages
                tail.insert(0, self._messages[-(tail_len + 1)])
                
        # Build body text to summarize
        body_msgs = self._messages[len(head):-len(tail)] if tail else self._messages[len(head):]
        if not body_msgs:
            return  # Nothing to summarize
            
        combined = "\n\n".join(
            f"{m.role}: {m.content or ''}"
            for m in body_msgs
            if m.content  # Skip empty content
        )
        
        # Ask model for summary
        summary_prompt = (
            "Summarize the following conversation so another agent could "
            "continue where we left off. Preserve any filenames, decisions, "
            "plans or variable names that matter.\n\n" + combined
        )
        
        # Get summary from model
        summary_msg = model.chat(
            messages=[{"role": "system", "content": summary_prompt}],
            tools=None
        )
        summary = summary_msg.content.strip()
        
        # Replace memory with head + summary + tail
        self._messages = head + [ChatMessage.assistant(summary)] + tail

###############################################################################
# OpenAI wrapper
###############################################################################


class LLMClient:
    """Thin wrapper around *openai* Chat Completions API."""

    def __init__(
        self,
        model_id: str = "gpt-4.1",
        temperature: float = 0.7,
        api_key: str | None = None,
        api_base: str | None = None,
        debug: bool = False,
    ) -> None:
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"), base_url=api_base or "https://api.openai.com/v1")
        self.model_id = model_id
        self.temperature = temperature
        self.debug = debug
        if self.model_id == 'o3':
            self.temperature = 1.0

    # ------------------------------------------------------------------
    def chat(self, *, messages: list[Mapping[str, Any]], tools: list[Mapping[str, Any]] | None = None) -> Any:  # noqa: D401
        if self.debug:
            LOGGER.info("Sending request → %s", self.model_id)
        resp = self.client.chat.completions.create(model=self.model_id, messages=messages, tools=tools or None,
                                                   tool_choice="auto" if tools else None, temperature=self.temperature)
        # Attach usage information to the message object so it's available for token counting display
        message = resp.choices[0].message
        if hasattr(resp, 'usage'):
            message.usage = resp.usage
        return message

###############################################################################
# Regex patterns to salvage malformed tool calls
###############################################################################

_PAT_BRACKETS = re.compile(r"\[TOOL_REQUEST\](.*?)\[END_TOOL_REQUEST\]", re.DOTALL | re.IGNORECASE)
_PAT_XML = re.compile(r"<TOOL\b[^>]*>(.*?)</TOOL>", re.DOTALL | re.IGNORECASE)
_PAT_FENCE = re.compile(r"```(?:tool|json)\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
_PAT_FUNC = re.compile(r"TOOL_CALL\s*\(\s*name\s*=\s*['\"]?([\w\-]+)['\"]?\s*,\s*args\s*=\s*(\{.*?\})\s*\)", re.DOTALL | re.IGNORECASE)
_PAT_BARE_JSON = re.compile(r"(?<![\w\-\"])(\{\s*\"name\"\s*:\s*\".+?\".+?\"arguments\"\s*:\s*\{.*?\}\s*})", re.DOTALL)
_PATTERNS = [_PAT_BRACKETS, _PAT_XML, _PAT_FENCE, _PAT_FUNC, _PAT_BARE_JSON]


def _json_from_blob(blob: str) -> dict[str, Any]:  # noqa: D401
    """Return the first valid JSON object embedded anywhere inside *blob*."""
    opens = [m.start() for m in re.finditer(r"{", blob)]
    for start in opens:
        depth = 0
        for idx in range(start, len(blob)):
            ch = blob[idx]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = _RE_TRAILING_COMMA.sub(r"\1", blob[start : idx + 1])
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break
    raise ValueError("no valid JSON found in blob")

###############################################################################
# The Agent class
###############################################################################


class Agent:
    """An autonomous coding agent that interacts with the OpenAI Chat API."""

    # Shared master-step counter ------------------------------------------------
    _global_step: int = 0

    def __init__(
        self,
        *,
        tools: Sequence[Tool],
        model: LLMClient,
        name: str = "agent",
        description: str = "Coding agents that completes tasks using tools.",
        max_steps: int = 20,
        verbosity: int = 1,
        planning_interval: int | None = None,
        memory_threshold: int | None = None,
        managed_agents: Sequence["Agent"] | None = None,
        add_tools_to_system_prompt: bool = True,
        clear_memory_on_run: bool = False,
        system_message: str = (
            "You are a highly skilled coding agent.  Your job is to complete "
            "the tasks assigned to you using the provided tools. When completely "
            "If the task is non-trivial start with `make_plan`.  Use `final_answer` *only* "
            "once everything is complete."
            "If you find yourself struggling with the same problem a couple times, "
        ),
    ) -> None:
        self.name = name
        self.description = description or ""
        # tools ----------------------------------------------------------
        self.tools: dict[str, Tool] = {t.name: t for t in tools}
        # ensure final_answer is always present
        if "final_answer" not in self.tools:
            self.tools["final_answer"] = FinalAnswer()
        # managed agents (need this before building system prompt) -------
        self.managed_agents: dict[str, Agent] = {a.name: a for a in managed_agents or []}
        # system prompt --------------------------------------------------
        self.system_prompt: str = self._build_system_prompt(system_message, add_tools_to_system_prompt)
        # model & memory -------------------------------------------------
        self.model = model
        self.memory = ConversationMemory()
        self.memory.append(ChatMessage.system(self.system_prompt))
        # misc config ----------------------------------------------------
        self.max_steps = max_steps
        self.verbosity = verbosity
        self.planning_interval = planning_interval
        self.memory_threshold = memory_threshold
        self.clear_memory_on_run = clear_memory_on_run
        
        # Add inputs/output_type for use as a tool (when this agent is called by another agent)
        self.inputs = {
            "task": {
                "type": "string", 
                "description": f"Task for {self.name} to execute and report back on."
            }
        }
        self.output_type = "string"
        
        self.tools.update(self.managed_agents)  # treat them like tools
        # logging --------------------------------------------------------
        self._trace_dir = f"agent_traces_{_dt.datetime.now():%y%m%d_%H%M}"
        os.makedirs(self._trace_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, task: str, *, reset: bool = None) -> str:  # noqa: D401
        """Run the agent until it produces a final answer."""
        # Use the attribute value if reset parameter is not provided
        should_reset = self.clear_memory_on_run if reset is None else reset
        if should_reset:
            self.memory = ConversationMemory()
            self.memory.append(ChatMessage.system(self.system_prompt))
        self.memory.append(ChatMessage.user(task))
        self._log("Task received", level=1)

        for local_step in range(1, self.max_steps + 1):
            self._increment_master_step()
            if self.planning_interval and (local_step == 1 or (local_step - 1) % self.planning_interval == 0):
                plan_prompt = ("Draft a plan for {{ task }}" if local_step == 1 else "Plan update - {{ remaining_steps }} steps left.")
                self.memory.append(ChatMessage.user(self._populate(plan_prompt, task=task, remaining_steps=self.max_steps - local_step)))
                _ = self._take_action()
            answer = self._take_action()
            
            # Write memory trace to file if verbosity level is high enough
            if self.verbosity >= 3:
                self._dump_trace()
            
            # Auto-summarize memory if it exceeds threshold
            if self.memory_threshold and len(self.memory) > self.memory_threshold:
                self._log(f"Memory threshold reached ({len(self.memory)} > {self.memory_threshold}), summarizing...", 2)
                self.memory.summarize(self.model)
                self._log(f"Memory summarized to {len(self.memory)} messages", 2)
                
            if answer is not None:
                self._log("Final answer produced", 1)
                return answer
        # ------------- exhausted budget - force summary ------------------
        self._log("Step budget exhausted - forcing summary", 1)
        summary = self._summarize_for_final()
        return summary

    # Let an Agent instance behave like a tool ------------------------------
    def __call__(self, *, task: str, reset: bool = None) -> str:  # noqa: D401
        return self.run(task, reset=reset)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @classmethod
    def _increment_master_step(cls) -> None:  # noqa: D401
        cls._global_step += 1

    def _log(self, msg: str, level: int = 1) -> None:  # noqa: D401
        if self.verbosity >= level:
            LOGGER.info("#%04d %s - %s", self._global_step, self.name, msg)

    # ------------------------------------------------------------------
    def _append_to_summary(self, tool_name: str, args: dict) -> None:  # noqa: D401
        """Append a line to the summary.txt file showing the agent action."""
        os.makedirs(self._trace_dir, exist_ok=True)
        summary_file = os.path.join(self._trace_dir, "summary.txt")
        
        # Format arguments for display
        args_str = ""
        if args:
            # Format args compactly or truncate if too long
            if len(str(args)) > 50:
                # For longer args, just show key info if possible
                if 'filename' in args:
                    arg_preview = f'filename="{args["filename"]}"'
                    if 'code' in args and len(str(args['code'])) > 20:
                        arg_preview += ',code=".."'
                    args_str = f"({arg_preview})"
                else:
                    args_str = "(...)"
            else:
                # For shorter args, include everything
                args_parts = []
                for k, v in args.items():
                    if isinstance(v, str):
                        args_parts.append(f'{k}="{v}"')
                    else:
                        args_parts.append(f'{k}={v}')
                args_str = f"({', '.join(args_parts)})"
        
        # Create the summary line
        summary_line = f"[{self._global_step}] {self.name} > {tool_name}{args_str}\n"
        
        # Append to the summary file
        with open(summary_file, "a", encoding="utf-8") as f:
            f.write(summary_line)
            
    def _dump_trace(self) -> None:  # noqa: D401
        """Write memory trace to a file with global step counter and agent name."""
        os.makedirs(self._trace_dir, exist_ok=True)
        fname = os.path.join(self._trace_dir, f"step_{self._global_step:03d}_{self.name}.log")
        with open(fname, "w", encoding="utf-8") as f:
            for idx, msg in enumerate(self.memory, 1):
                role = msg.role.upper()
                content = msg.content or ""
                
                # Write separator between messages
                separator = "-" * 80
                if idx > 1:
                    f.write(f"\n{separator}\n\n")
                
                if msg.role == "assistant":
                    f.write(f"[{idx}] {role} [{self.name}]:\n{content}\n")
                    if msg.tool_calls:
                        f.write("TOOL CALLS:\n")
                        f.write("[\n")
                        for tc in msg.tool_calls:
                            f.write(f"  {{\n")
                            f.write(f"    \"id\": \"{tc.get('id', '')}\",\n")
                            f.write(f"    \"type\": \"function\",\n")
                            f.write(f"    \"function\": {{\n")
                            f.write(f"      \"name\": \"{tc['function']['name']}\",\n")
                            
                            # Format arguments for better display
                            args = tc['function'].get('arguments', '{}')
                            
                            # Parse and format arguments
                            try:
                                # Handle both string and dict forms
                                if isinstance(args, str):
                                    args_obj = json.loads(args)
                                else:
                                    args_obj = args
                                
                                # Wrap long text values
                                if isinstance(args_obj, dict):
                                    for key, value in args_obj.items():
                                        if isinstance(value, str) and len(value) > 100:
                                            args_obj[key] = textwrap.fill(value, width=80)
                                
                                # Format with indentation and remove first level
                                formatted_args = json.dumps(args_obj, indent=6, ensure_ascii=False)
                                formatted_args = "\n".join(
                                    line[2:] if line.startswith("  ") else line 
                                    for line in formatted_args.split("\n")
                                )
                                
                                f.write(f"      \"arguments\": {formatted_args}\n")
                            except Exception:
                                # Fallback for any formatting errors
                                f.write(f"      \"arguments\": \"{str(args)}\"\n")
                            
                            f.write(f"    }}\n")
                            f.write(f"  }}\n")
                        f.write("]\n")
                                            
                    # Add token count if available
                    if hasattr(msg, 'usage') and msg.usage:
                        tokens_used = getattr(msg.usage, 'total_tokens', 'unknown')
                        f.write(f"[TOKENS: {tokens_used}]\n")
                    
                elif msg.role == "tool":
                    f.write(f"[{idx}] TOOL RESPONSE from {msg.name}:\n{content}\n")
                else:
                    f.write(f"[{idx}] {role}:\n{content}\n")
        self._log(f"Trace dumped to {fname}", 3)

    # ------------------------------------------------------------------
    def _build_system_prompt(self, base: str, add_tools: bool) -> str:  # noqa: D401
        if not add_tools:
            return base
        lines = [base, "", "Here are your tools:"]
        
        # Regular tools first
        for tool in self.tools.values():
            # Skip managed agents - they'll be handled separately
            if tool in self.managed_agents.values():
                continue
            inp = ", ".join(f"{k}: {v['type']}" for k, v in tool.inputs.items()) or "None"
            lines.append(f"- {tool.name}: {tool.description} (inputs: {inp})")
        
        # Add managed agents section if we have any
        if self.managed_agents:
            lines.append("\nYou can also give tasks to team members:")
            lines.append("Calling a team member works the same as calling a tool: the only argument you need to provide is 'task', a string explaining what you want them to do.")
            lines.append("Since these team members are specialized agents, be clear and detailed in your task descriptions.")
            for agent in self.managed_agents.values():
                description = getattr(agent, 'description', '')
                if getattr(agent, 'clear_memory_on_run', False):
                    description += " This team member starts with a clean memory for each task and needs comprehensive context."
                lines.append(f"- {agent.name}: {description}")
        
        lines.append("\nCall tools by returning a message that contains *only* a valid tool-call JSON object.")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def _take_action(self) -> str | None:  # noqa: D401 - returns final answer or None
        """Send current memory to LLM, execute any tool calls, append responses."""
        # ---- call the model -------------------------------------------
        msg = self.model.chat(messages=self.memory.to_openai(), tools=[self._tool_to_openai(t) for t in self.tools.values()])
        response_content = msg.content
        # tool_calls = getattr(msg, "tool_calls", None)
        tool_calls = _normalise_tool_calls(getattr(msg, "tool_calls", None))
        if tool_calls is None:  # fallback to regex salvage
            maybe_calls = self._extract_tool_calls(response_content or "")
            if maybe_calls:
                tool_calls = maybe_calls
                response_content = None
        assistant_msg = ChatMessage.assistant(response_content, tool_calls=tool_calls)
        # Transfer usage information from API response to our ChatMessage object
        if hasattr(msg, 'usage'):
            assistant_msg.usage = msg.usage
        self.memory.append(assistant_msg)
        if self.verbosity >= 2 and _console is not None:
            _console.print(assistant_msg._pretty())
            if hasattr(msg, 'usage') and msg.usage:
                tokens_used = getattr(msg.usage, 'total_tokens', 'unknown')
                _console.print(f"[dim]Tokens used this step: {tokens_used}[/dim]")
        # ---- dispatch tool calls --------------------------------------
        if not tool_calls:  # No tool was called - instruct agent to use a tool
            self._log("No tool call used, instructing the agent to try again", 1)
            msg = "You must use a tool call. Use final_answer if you are FINISHED, otherwise use a different tool."
            self.memory.append(ChatMessage.user(msg))
            return None
            
        # Dump trace before processing any tool calls to capture pre-execution state
        if self.verbosity >= 3:
            self._dump_trace()
            
        final_answer: str | None = None
        image_parts: list[dict[str, Any]] = []
        for tc in tool_calls:
            name = tc["function"]["name"]
            raw_args = tc["function"].get("arguments", {})
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            target = self.tools.get(name)
            if target is None:
                err = f"Unknown tool '{name}'"
                self.memory.append(ChatMessage.tool(name=name, tool_call_id=tc.get("id", "call_0"), result=err))
                continue
                
            # Log this tool call to the summary file
            self._append_to_summary(name, args)
            try:
                result = target(**args)
            except Exception as exc:  # pragma: no cover - runtime errors
                result = f"ToolError[{name}]: {exc} ({traceback.format_exc().splitlines()[-1]})"
            
            content_for_log = str(result)

            # Capture any image data-URI returned by tools
            if (getattr(target, "output_type", "") == "image"
                and isinstance(result, str)
                and result.startswith("data:image")):
                image_parts.append(
                    {"type": "image_url",
                    "image_url": {"url": result, 
                                  "detail": "auto"}} # or low/high
                )
                basename = os.path.basename(args.get("filename", "image"))
                content_for_log = f"<{basename} • {len(result):,} chars>"

            self.memory.append(ChatMessage.tool(
                name=name,
                tool_call_id=tc.get("id", "call_0"),
                result=content_for_log))
                
            if name == "final_answer":
                final_answer = args.get("answer", str(result))
            if self.verbosity >= 2 and _console is not None:
                _console.print(self.memory[-1]._pretty())

            # Feed images back to the model so it can "see" them
            if image_parts:
                self.memory.append(ChatMessage.user(image_parts))

        return final_answer

    # ------------------------------------------------------------------
    def _summarize_for_final(self) -> str:  # noqa: D401
        """Force a summary via *final_answer* tool after step budget exhausted."""
        summary_prompt = "Summarise the current progress so the user can continue on their own."
        # Only *final_answer* tool is allowed
        msg = self.model.chat(messages=self.memory.to_openai() + [ChatMessage.user(summary_prompt).to_openai()],
                               tools=[self._tool_to_openai(self.tools["final_answer"])])
        tool_calls = _normalise_tool_calls(getattr(msg, "tool_calls", None))
        if tool_calls is None:
            return msg.content
        tc = tool_calls[0] #if getattr(msg, "tool_calls", None) else None
        if tc and tc["function"]["name"] == "final_answer":
            answer = json.loads(tc["function"].get("arguments", "{}")).get("answer", "")
            return answer
        # # Fallback: return raw content
        # return msg.content or "(no summary)"

    # ------------------------------------------------------------------
    @staticmethod
    def _tool_to_openai(tool: Tool) -> dict[str, Any]:  # noqa: D401
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": {k: {"type": v["type"], "description": v["description"]} for k, v in tool.inputs.items()},
                    "required": list(tool.inputs.keys()),
                },
            },
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _extract_tool_calls(text: str) -> list[dict[str, Any]]:  # noqa: D401
        calls: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        spans: list[tuple[int, int]] = []
        for pat in _PATTERNS:
            for m in pat.finditer(text):
                start, end = m.span()
                if any(max(start, s) < min(end, e) for s, e in spans):
                    continue
                try:
                    data = _json_from_blob(m.group(1)) if pat is not _PAT_FUNC else {"name": m.group(1), "arguments": _json_from_blob(m.group(2))}
                    name = data.get("name") or data.get("tool") or data.get("function", {}).get("name")
                    args = data.get("arguments") or data.get("args") or data.get("function", {}).get("arguments", {})
                    args_str = args if isinstance(args, str) else json.dumps(args)
                    key = (name, args_str)
                    if key in seen:
                        continue
                    spans.append((start, end))
                    seen.add(key)
                    calls.append({"id": f"local_{uuid.uuid4().hex[:8]}", "type": "function", "function": {"name": name, "arguments": args_str}})
                except Exception:  # pragma: no cover - best effort salvage
                    continue
        return calls

    # ------------------------------------------------------------------
    @staticmethod
    def _populate(tmpl: str, **vars: Any) -> str:  # noqa: D401
        out = tmpl
        for k, v in vars.items():
            out = out.replace(f"{{{{ {k} }}}}", str(v)).replace(f"{{{{{k}}}}}", str(v))
        return out

###############################################################################
# CLI convenience entry-point
###############################################################################


def _build_default_agent(debug: bool, local: bool, confirm_edits: bool = False, oai_model: str = 'gpt-4.1') -> Agent:  # noqa: D401
    """Build a default agent with standard tools.
    
    Args:
        debug: Enable verbose logging of API requests/responses
        local: Use a local LLM server instead of OpenAI API
        confirm_edits: Whether file edits and deletes require user confirmation
    """
    model = LLMClient(model_id="lmstudio" if local else oai_model, api_base="http://localhost:1234/v1" if local else None, debug=debug)
    tools: list[Tool] = [
        WriteFile(), 
        ReadFile(), 
        EditFile(confirm_edits=confirm_edits),  # Pass configuration to tools
        Delete(confirm_edits=confirm_edits),    # Pass configuration to tools
        RunPython(), 
        RunBash(), 
        ViewImage(),
        ListFiles(), 
        MakePlan(), 
        GetUserInput(),
        FinalAnswer()
    ]
    return Agent(tools=tools, model=model, max_steps=25, verbosity=3, name="code_agent", description="Writes/tests Python projects")


def main() -> None:  # noqa: D401
    parser = argparse.ArgumentParser(description="Run the autonomous Agent")
    parser.add_argument("task", nargs="?", help="Initial user task (prompted if omitted)")
    parser.add_argument("-d", "--debug", action="store_true", help="Verbose OpenAI request/response logging")
    parser.add_argument("-l", "--local", action="store_true",
                        help="Use a local LLM instead of the OpenAI API for the executor agent")
    parser.add_argument("-v", "--verbosity", type=int, default=3, choices=range(0,4),
                        help="Verbosity level: 0-quiet 1-steps 2-rich 3-trace (default 2)")
    parser.add_argument("-m", "--model", default="gpt-4.1", help="OpenAI model")
    parser.add_argument("-w", "--wkdir", action="store_true", help="move to work dir")
    parser.add_argument("-c", "--confirm-edits", action="store_true", 
                        help="Require confirmation before editing or deleting files")
    parser.add_argument("--confirm-plan", action="store_true", 
                        help="Ask for confirmation after making initial plan")
    parser.add_argument( "--end", action="store_true", 
                        help="End on first final_answer")
    parser.add_argument("--multi", action="store_true",
                        help="Use multiple agents starting with a manager")
    parser.add_argument( "--vision", action="store_true", 
                        help="The model has vision and can use view_image tool")

    args = parser.parse_args()



    # If task ends with .txt, assume it's a file path and read from it
    if args.task and args.task.endswith(".txt"):
        user_task = open(args.task).read()
    elif args.task:
        user_task = args.task
    else:
        # Fall back to prompting if no task provided
        user_task = input("Enter your task: ")


    if args.wkdir:
        cwd = None
        if cwd is None:
            cwd = os.getcwd()

        workdir = os.path.join(cwd,'work/tmp_'+_dt.datetime.now().strftime('%y%m%d_%H%M'))
        os.makedirs(workdir,exist_ok=True)
        print(f'moving to {workdir}')
        os.chdir(workdir)

    tools_all = [
        WriteFile(), ReadFile(), EditFile(confirm_edits=args.confirm_edits), RunPython(),
        RunBash(), Delete(confirm_edits=args.confirm_edits),
        MakePlan(), ListFiles(), FinalAnswer()
    ]
    if args.vision:
        tools_all.insert(-1,ViewImage())

    model = LLMClient(
        model_id="lmstudio" if args.local else args.model,
        debug=args.debug,
        api_base="http://localhost:1234/v1" if args.local else None,
    )

    agent = Agent(tools=tools_all,
                    model=model,
                    max_steps=25,
                    verbosity=args.verbosity,
                    name="code_agent",
                    description="Writes/tests Python projects")
    if args.multi:
        agent_code = agent
        agent_code.clear_memory_on_run = True

        manager_prompt = ("You are the *manager_agent* - a senior engineer. Your first "
            "task is to use the make_plan tool. In the plan you should start with "
            "Goal: <the overall goal of the task -- restating it in your own words> "
            "Completion Criteria: <what did the user specify that woiuld complete the task> "
            "Next, break the user's high-level request into a numbered sequence of "
            "concrete, executable steps. Each step MUST include: "
            "(1) a clear instruction for the code_agent "
            "(2) any parameters or file names needed, and "
            "(3) explicit completion criteria. You are the expert in this area, "
            "so be very clear about the details and instructions so the code_agent doesn't "
            "have to fill in too many blanks. "
            "Your plan should not be overly complex -- accomplish the task in the minimal "
            "number of steps necessary. Do NOT ask the agent to write scaffolding files first. " 
            "Do NOT ask the agent not use a virtual environment or git repo or install anyything."
            ""
            "Then you must delegate the *first* step to `code_agent` and "
            "wait for its report. When a step is reported complete, mark it as "
            "done ✅ and delegate the next. Repeat until all steps are finished. "
            "For the Second step and beyond, the code_agent does not know of any "
            "previous work so be sure to give complete and verbose context. "
            "NEVER call final_answer until all steps are reported ✅ complete by "
            "code_agent. If you think work is finished, first verify each "
            "criterion, then call final_answer.")

        tools_manager = [
            MakePlan(), FinalAnswer()
        ]
        if args.confirm_plan:
            tools_manager.append(GetUserInput())

        agent = Agent(tools=tools_manager,
                        model=model,
                        system_message=manager_prompt,
                        managed_agents=[agent_code],
                        max_steps=25,
                        verbosity=args.verbosity,
                        name="manager_agent",
                        description="Magnages coding agents")
    
        if args.confirm_plan:
                manager_prompt+="\n\nAfter creating the plan, ask the user for any further changes or approval."

        # agent = _build_default_agent(debug=args.debug,
        #                             local=args.local,
        #                             confirm_edits=args.confirm_edits,
        #                             oai_model=args.model)

        
    result = agent.run(user_task)
    print("\n=== FINAL ANSWER ===\n", result)

    # optional interactive loop -------------------------------------------
    while True and not args.end:
        follow = input("Feedback (or 'end'): ").strip()
        if follow.lower() == "end":
            break
        agent.max_steps += 20  # give more budget
        print(agent.run(follow, reset=False))

    if args.wkdir:
        os.chdir(cwd)


if __name__ == "__main__":  # pragma: no cover
    main()
