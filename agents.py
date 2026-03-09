"""Autonomous coding agents with tool use, memory, and a CLI entrypoint."""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import re
import shutil
import textwrap
import time
import traceback
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from agent_tools import (
    Tool, WriteFile, ReadFile, EditFile, RunPython, Shell, ViewImage,
    ListFiles, SearchFiles, MakePlan, Reflect, Commentary, FinalAnswer, GetUserInput,
    UpdatePlan, ReadPDF, _RE_TRAILING_COMMA, _parse_command_result
)

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.text import Text

_console: Console | None = Console()

from openai import OpenAI

LOGGER = logging.getLogger("agent")
logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s ▸ %(message)s")

_LEXER_BY_EXT = {
    ".bash": "bash",
    ".c": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".css": "css",
    ".diff": "diff",
    ".go": "go",
    ".html": "html",
    ".java": "java",
    ".js": "javascript",
    ".json": "json",
    ".md": "markdown",
    ".patch": "diff",
    ".py": "python",
    ".rb": "ruby",
    ".rs": "rust",
    ".sh": "bash",
    ".sql": "sql",
    ".toml": "toml",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".txt": "text",
    ".xml": "xml",
    ".yaml": "yaml",
    ".yml": "yaml",
}

_LIVE_DEPTH = 0


def _clip(text: Any, max_length: int = 100) -> str:
    rendered = str(text).strip().replace("\n", " ")
    if len(rendered) <= max_length:
        return rendered
    return rendered[: max_length - 3] + "..."


def _console_input(prompt: str, console: Console | None = None) -> str:
    console = _console if console is None else console
    if console is not None:
        return console.input(prompt)
    return input(prompt)


def _console_print(
    renderable: Any = "",
    *,
    style: str | None = None,
    console: Console | None = None,
) -> None:
    console = _console if console is None else console
    if console is not None:
        if isinstance(renderable, str) and style:
            console.print(Text(renderable, style=style))
        else:
            console.print(renderable)
        return
    if isinstance(renderable, Text):
        print(renderable.plain)
    else:
        print(renderable)


def _syntax_lexer(path: str | None) -> str:
    if not path:
        return "text"
    return _LEXER_BY_EXT.get(os.path.splitext(path)[1].lower(), "text")


def _render_final_answer_panel(answer: str, title: str = "Final Answer") -> None:
    body = Text(answer or "", style="white")
    _console_print(Panel(body, title=title, border_style="green", expand=True))


def _format_elapsed(seconds: float | int | None) -> str:
    if seconds is None:
        return "0s"
    total = max(0, int(seconds))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"
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
    role: str
    content: str | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: Any | None = None
    usage: Any | None = None

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

    def to_session(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "name": self.name,
            "tool_call_id": self.tool_call_id,
            "tool_calls": _normalise_tool_calls(self.tool_calls),
        }

    @classmethod
    def from_session(cls, data: Dict[str, Any]) -> "ChatMessage":
        role = data.get("role")
        content = data.get("content")
        if role == "assistant":
            return cls.assistant(content, tool_calls=data.get("tool_calls"))
        if role == "tool":
            return cls.tool(
                name=data.get("name", ""),
                tool_call_id=data.get("tool_call_id", ""),
                result=content or ""
            )
        if role == "system":
            return cls.system(content or "")
        if role == "user":
            return cls("user", content)
        return cls(role or "user", content)


class ConversationMemory:
    """Conversation history with optional automatic summarisation."""

    def __init__(self) -> None:
        self._messages: list[ChatMessage] = []

    def __iter__(self):
        return iter(self._messages)

    def __len__(self) -> int:  # noqa: D401
        return len(self._messages)

    def append(self, value: ChatMessage) -> None:
        self._messages.append(value)

    def to_openai(self) -> List[Dict[str, Any]]:  # noqa: D401
        return [m.to_openai() for m in self._messages]

    def summarize(self, model: Any, tail_len: int = 0) -> None:
        if len(self._messages) <= 6:
            return

        head = list(self._messages[:2])
        tail: list[ChatMessage] = []
        if tail_len > 0:
            tail = list(self._messages[-tail_len:])
            while tail and tail[0].role == "tool":
                tail.pop(0)
                if len(tail) < 2 and len(self._messages) > len(head) + 2:
                    tail.insert(0, self._messages[-(tail_len + 1)])
        body_msgs = self._messages[len(head):-len(tail)] if tail else self._messages[len(head):]
        if not body_msgs:
            return
        combined = "\n\n".join(
            f"{m.role}: {m.content or ''}"
            for m in body_msgs
            if m.content
        )
        summary_instruction = (
            "You summarize conversations for agent handoff. "
            "Be complete and factual, and preserve filenames, decisions, plans, "
            "variable names, failed attempts, and directories in use."
        )
        summary_prompt = (
            "Summarize the following conversation so another agent can "
            "continue where we left off. Be absolutely complete so nothing is "
            "forgotten.\n\nConversation:\n" + combined
        )
        summary_msg = model.chat(
            messages=[
                {"role": "system", "content": summary_instruction},
                {"role": "user", "content": summary_prompt},
            ],
            tools=None
        )
        summary = textwrap.dedent(
            f"""\
            Below is a summary of the previous conversation and work completed so far.
            Use it as context and continue from this state instead of restarting the task.

            ######
            {summary_msg.content.strip()}
            ######

            Remember:
            - Continue after the latest user instruction.
            - When taking actions, use tool calls instead of only describing the tool you want to use.
            - If the task is complete, call `final_answer` instead of only saying that you are finished.
            """
        ).strip()
        self._messages = head + [ChatMessage.user(summary)] + tail

###############################################################################
# OpenAI wrapper
###############################################################################


class LLMClient:
    """Thin wrapper around *openai* Chat Completions API."""

    def __init__(
        self,
        model_id: str = "gpt-5.4",
        temperature: float = 0.2,
        api_key: str | None = None,
        api_base: str | None = None,
        debug: bool = False,
    ) -> None:
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"), base_url=api_base or "https://api.openai.com/v1")
        self.model_id = model_id
        self.temperature = temperature
        self.debug = debug
        if self.model_id.startswith('o') or  self.model_id.startswith('gpt-5'):
            self.temperature = 1.0

    # ------------------------------------------------------------------
    def chat(self, *, messages: list[Mapping[str, Any]], tools: list[Mapping[str, Any]] | None = None) -> Any:  # noqa: D401
        if self.debug:
            LOGGER.info("Sending request → %s", self.model_id)
        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "temperature": self.temperature,
        }
        if tools: # only include tool-related params when we actually have tools.
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        resp = self.client.chat.completions.create(**kwargs)
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
_PAT_TOOL_CALLS_SPECIAL = re.compile(r"\[TOOL_CALLS\](\w+)<[^>]*>(\{.*?\})", re.DOTALL | re.IGNORECASE) # used by mistral_small sometimes (SPECIAL_32)
_PAT_TOOL_CALLS_DIRECT = re.compile(r"\[TOOL_CALLS\](\w+)(\{.*?\})", re.DOTALL | re.IGNORECASE)
_PAT_TOOL_CALLS_ARGS = re.compile(r"\[TOOL_CALLS\](\w+)\[ARGS\](\{.*)", re.DOTALL | re.IGNORECASE)
_PAT_GPT_OSS = re.compile(r"<\|channel\|>commentary to=functions\.(\w+)\s*<\|constrain\|>json<\|message\|>(\{.*\})", re.DOTALL | re.IGNORECASE)
_PATTERNS = [_PAT_BRACKETS, _PAT_XML, _PAT_FENCE, _PAT_FUNC, _PAT_BARE_JSON, _PAT_TOOL_CALLS_SPECIAL, _PAT_TOOL_CALLS_DIRECT, _PAT_TOOL_CALLS_ARGS, _PAT_GPT_OSS]


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
        verbosity: int = 2,
        auto_save: bool = True,
        trace_enabled: bool = True,
        planning_interval: int | None = None,
        memory_threshold: int | None = None,
        managed_agents: Sequence["Agent"] | None = None,
        add_tools_to_system_prompt: bool = True,
        clear_memory_on_run: bool = False,
        include_function_thoughts: bool = True,
        system_message: str = textwrap.dedent('''
            You are a precise coding agent working in a terminal workspace.
            Complete the user's task using the available tools.

            Core behavior:
            - Keep going until the task is fully resolved or you are truly blocked.
            - Do not guess about code, files, or results. Read, run, inspect, and verify.
            - If you say you will take an action, take it with a tool call.
            - If you decide a tool call is needed, actually make the tool call instead of only describing it.
            - Respect the existing codebase. Be surgical in established projects.
            - Fix root causes when practical; avoid cosmetic or unrelated changes.
            - Prefer simple, robust solutions over clever or sprawling ones.
            - Work step by step and do not end your turn until you are sure the task is solved or you are genuinely blocked.
            - Take your time to reason carefully, but keep making progress toward a verified solution.

            Planning:
            - Use update_plan for non-trivial, ambiguous, or multi-step work.
            - For any task likely to take more than a couple of meaningful actions, call update_plan early, usually within the first 1-3 tool calls.
            - Skip update_plan for simple one-step tasks that can be completed immediately.
            - Plans should be short, concrete, easy to verify, and free of padding.
            - Prefer 3-6 action-oriented steps rather than vague or padded plans.
            - Keep exactly one step in_progress until everything is done.
            - Update the plan when a step starts, completes, stalls, or when the approach changes.
            - If you finish a meaningful piece of work and have not updated the plan yet, update it before moving on.
            - Before final_answer, make sure the plan accurately reflects what is complete and what remains.
            - Do not repeat the full plan in prose after calling update_plan; the UI already shows it.
            - Restate the completion criteria to yourself before implementation and again before final_answer.

            Execution:
            - First understand the task, constraints, and completion criteria.
            - Gather the minimum context needed before editing.
            - Work incrementally: inspect, edit, run targeted checks, then refine.
            - Prefer focused edits over rewriting entire files unless a rewrite is clearly cleaner.
            - Do not create unnecessary versioned files like *_v2.py or *_final.py.
            - Reuse existing project patterns, naming, and structure.
            - Do not chase unrelated bugs, failing tests, or cleanup outside the task.
            - Avoid re-reading files you just wrote or patched unless something external may have changed them or you need exact lines.
            - Prefer small, testable changes over sprawling rewrites.

            Debugging:
            - When behavior is wrong, identify the root cause before patching.
            - Form a concrete hypothesis, test it, and use the result to choose the next step.
            - Preserve useful observations from failures and use them to narrow the search.
            - Revisit assumptions quickly when results do not match expectations.
            - Use targeted debugging aids when needed to inspect state and validate hypotheses, then clean them up when finished.

            Validation:
            - Validate changed behavior whenever practical.
            - Start with the most targeted check you can run, then broaden if needed.
            - If the repo has tests or build commands relevant to the change, use them.
            - If there is no test harness, do not invent a large one just for this task.
            - For scientific, numerical, or plotting work, run the code and check whether outputs are physically or numerically plausible.
            - Inspect generated images or reports when they matter to correctness.
            - Operate headlessly; do not rely on interactive plot windows like matplotlib show().
            - Do not claim to have run or verified anything you did not actually run or verify.
            - After you believe you are done, verify the completion criteria one by one.

            Approvals and blockers:
            - Some tool calls may require user approval. If a change is rejected, absorb the feedback and continue.
            - If information is missing and cannot be inferred safely, ask only the minimum necessary question.
            - If blocked by a hard constraint, explain it briefly and propose the best next action.

            Final answer:
            - Use final_answer only when the task is complete or you are genuinely blocked.
            - Before final_answer, restate the completion criteria mentally and check them one by one.
            - Final responses should be concise, concrete, and honest about verification and remaining caveats.
            '''),
        ) -> None:
        self.name = name
        self.description = description or ""
        self.tools: dict[str, Tool] = {t.name: t for t in tools}
        if "final_answer" not in self.tools:
            self.tools["final_answer"] = FinalAnswer()
        self.managed_agents: dict[str, Agent] = {a.name: a for a in managed_agents or []}
        self.system_prompt: str = self._build_system_prompt(system_message, add_tools_to_system_prompt)
        self.model = model
        self.memory = ConversationMemory()
        self.memory.append(ChatMessage.system(self.system_prompt))
        self.max_steps = max_steps
        self.verbosity = verbosity
        self.trace_enabled = trace_enabled
        self.planning_interval = planning_interval
        self.memory_threshold = memory_threshold
        self.clear_memory_on_run = clear_memory_on_run
        self.include_function_thoughts = include_function_thoughts
        self._current_plan: dict[str, Any] | None = None
        self._run_started_at: float | None = None
        self._latest_prompt_tokens: int | None = None
        self._compact_single_agent_ui: bool = True
        self._current_local_step: int = 0
        self._status_message: str = "Idle"
        self._live: Live | None = None
        self._owns_live: bool = False
        self._task_preview: str = ""
        self.inputs = {"task": {"type": "string", "description": f"Task for {self.name} to execute and report back on."}}
        self.output_type = "string"
        self.tools.update(self.managed_agents)
        self._trace_dir = f"agent_traces_{_dt.datetime.now():%y%m%d_%H%M}"

    def run(self, task: str, *, reset: bool = None) -> str:  # noqa: D401
        """Run the agent until it produces a final answer."""
        should_reset = self.clear_memory_on_run if reset is None else reset
        if should_reset:
            self.memory = ConversationMemory()
            self.memory.append(ChatMessage.system(self.system_prompt))
            self._current_plan = None
        self.memory.append(ChatMessage.user(task))
        self._log("Task received", level=1)
        console = self._console_ref()
        clip_width = max(40, (console.width if console is not None else 100) - 20)
        self._task_preview = _clip(task, clip_width)
        self._run_started_at = time.perf_counter()
        self._latest_prompt_tokens = None
        self._current_local_step = 0
        self._status_message = "Thinking"
        self._start_live_display()
        try:
            if self.verbosity >= 1:
                self._ui_print(Rule(f"{self.name} [{getattr(self.model, 'model_id', 'model')}]"))
                self._render_event(self._task_preview, label="task", style="bright_black")
                self._refresh_live()

            for local_step in range(1, self.max_steps + 1):
                self._increment_master_step()
                self._current_local_step = local_step
                self._status_message = "Thinking"
                self._refresh_live()
                if self.planning_interval and (local_step == 1 or (local_step - 1) % self.planning_interval == 0):
                    answer = self._take_action(specific_tools=['update_plan'])
                else:
                    answer = self._take_action()
                if self.trace_enabled:
                    self._dump_trace()
                if self.memory_threshold and len(self.memory) > self.memory_threshold:
                    self._log(f"Memory threshold reached ({len(self.memory)} > {self.memory_threshold}), summarizing...", 2)
                    self.memory.summarize(self.model)
                    self._log(f"Memory summarized to {len(self.memory)} messages", 2)
                if answer is not None:
                    self._log("Final answer produced", 1)
                    self._status_message = "Final answer ready"
                    self._refresh_live()
                    return answer
            self._log("Step budget exhausted - forcing summary", 1)
            self._status_message = "Summarizing"
            self._refresh_live()
            summary = self._summarize_for_final()
            return summary
        finally:
            self._stop_live_display()

    def __call__(self, *, task: str, reset: bool = None) -> str:  # noqa: D401
        return self.run(task, reset=reset)

    @classmethod
    def _increment_master_step(cls) -> None:  # noqa: D401
        cls._global_step += 1

    def _log(self, msg: str, level: int = 1) -> None:  # noqa: D401
        if self.model.debug and self.verbosity >= level:
            LOGGER.info("#%04d %s - %s", self._global_step, self.name, msg)

    def _console_ref(self) -> Console | None:
        if self._live is not None:
            return self._live.console
        return _console

    def _ui_print(self, renderable: Any = "", *, style: str | None = None) -> None:
        _console_print(renderable, style=style, console=self._console_ref())

    def _ui_input(self, prompt: str) -> str:
        return _console_input(prompt, self._console_ref())

    def _build_plan_renderable(self) -> Panel | None:
        if not self._current_plan:
            return None
        explanation = str(self._current_plan.get("explanation") or "").strip()
        plan_items = self._current_plan.get("plan", []) or []
        body = Text()
        if explanation:
            body.append(explanation, style="bright_black")
            if plan_items:
                body.append("\n")
        markers = {
            "completed": ("[x] ", "green"),
            "in_progress": ("[>] ", "bold cyan"),
            "pending": ("[ ] ", "white"),
        }
        for idx, item in enumerate(plan_items):
            if idx:
                body.append("\n")
            status = str(item.get("status", "pending"))
            step = str(item.get("step", "")).strip()
            marker, style = markers.get(status, ("[ ] ", "white"))
            body.append(marker, style=style)
            body.append(step, style=style)
        if not body:
            body.append("No plan items yet.", style="bright_black")
        return Panel(body, title="Updated Plan", border_style="cyan", expand=True)

    def _build_status_renderable(self) -> Group:
        elapsed = _format_elapsed(
            None if self._run_started_at is None else time.perf_counter() - self._run_started_at
        )
        model_id = str(getattr(self.model, "model_id", "model"))
        token_text = (
            f"{self._latest_prompt_tokens:,} tok"
            if self._latest_prompt_tokens is not None
            else "-- tok"
        )
        cwd_text = os.getcwd()
        home = os.path.expanduser("~")
        if cwd_text.startswith(home):
            cwd_text = "~" + cwd_text[len(home):]
        step_text = f"step {self._current_local_step}/{self.max_steps}" if self._current_local_step else "step 0"
        status_value = self._status_message or "Working"
        status_text = Text()
        if status_value in {"Thinking", "Summarizing"}:
            frames = ("-", "\\", "|", "/")
            frame = frames[int(time.perf_counter() * 8) % len(frames)]
            status_text.append(f"{frame} ", style="cyan")
        status_text.append(_clip(status_value, 72), style="bold white")
        meta = Text()
        meta.append(model_id, style="bold bright_white")
        meta.append(" • ", style="bright_black")
        meta.append(token_text, style="cyan")
        meta.append(" • ", style="bright_black")
        meta.append(step_text, style="bright_black")
        meta.append(" • ", style="bright_black")
        meta.append(elapsed, style="bright_black")
        meta.append(" • ", style="bright_black")
        meta.append(_clip(cwd_text, 72), style="bright_black")
        return Group(status_text, meta)

    def _build_live_renderable(self) -> Group:
        return Group(self._build_status_renderable())

    def _refresh_live(self) -> None:
        if self._live is not None:
            self._live.refresh()

    def _start_live_display(self) -> None:
        global _LIVE_DEPTH
        if self.verbosity < 1 or _console is None or _LIVE_DEPTH > 0:
            self._live = None
            self._owns_live = False
            return
        self._live = Live(
            self._build_live_renderable(),
            console=_console,
            refresh_per_second=8,
            transient=False,
            vertical_overflow="visible",
            get_renderable=self._build_live_renderable,
        )
        self._live.start()
        self._owns_live = True
        _LIVE_DEPTH += 1

    def _stop_live_display(self) -> None:
        global _LIVE_DEPTH
        if self._live is not None and self._owns_live:
            self._live.stop()
            _LIVE_DEPTH = max(0, _LIVE_DEPTH - 1)
            if _console is not None:
                _console.print()
        self._live = None
        self._owns_live = False

    def _render_event(
        self,
        message: str,
        *,
        level: int = 1,
        label: str | None = None,
        style: str = "white",
    ) -> None:
        if self.verbosity < level:
            return
        compact_ui = self._compact_single_agent_ui and not self.managed_agents
        prefix = f"{self._global_step:04d} {self.name}" if label is None else label
        if compact_ui:
            prefix_width = max(16, len(prefix) + 2 if prefix else 16)
        else:
            default_prefix = f"{self._global_step:04d} {self.name}"
            prefix_width = max(16, len(default_prefix) + 2)
        prefix_block = f"{prefix:<{prefix_width}}" if prefix else " " * prefix_width
        wrapped_lines: list[str] = []
        console = self._console_ref()
        wrap_width = max(20, ((console.width if console is not None else 100) - prefix_width))
        for raw_line in str(message).splitlines() or [""]:
            pieces = textwrap.wrap(
                raw_line,
                width=wrap_width,
                replace_whitespace=False,
                drop_whitespace=False,
            )
            wrapped_lines.extend(pieces or [""])
        if console is None:
            lines = [
                (prefix_block if idx == 0 else " " * prefix_width) + line
                for idx, line in enumerate(wrapped_lines or [""])
            ]
            print("\n".join(lines))
            return
        rendered = Text()
        for idx, line in enumerate(wrapped_lines or [""]):
            rendered.append(prefix_block if idx == 0 else " " * prefix_width, style="bold bright_black")
            rendered.append(line, style=style)
            if idx < len(wrapped_lines) - 1:
                rendered.append("\n")
        console.print(rendered, soft_wrap=True)

    def _render_detail(self, detail: Mapping[str, Any] | None, *, force: bool = False) -> None:
        if not detail:
            return
        if not force and self.verbosity < 2:
            return

        kind = str(detail.get("kind", "text"))
        title = str(detail.get("title", "Details"))
        border_style = str(detail.get("border_style", "blue"))

        if kind == "diff":
            body = str(detail.get("body", ""))
            renderable = Syntax(body, "diff", theme="ansi_dark", word_wrap=True)
        elif kind == "code":
            body = str(detail.get("body", ""))
            path = detail.get("path")
            renderable = Syntax(body, _syntax_lexer(str(path) if path else None), theme="ansi_dark", word_wrap=False)
        elif kind == "json":
            rendered = json.dumps(detail.get("data", {}), indent=2, ensure_ascii=False)
            renderable = Syntax(rendered, "json", theme="ansi_dark", word_wrap=True)
        elif kind == "plan":
            renderable = self._build_plan_renderable() or Text(str(detail.get("body", "")), style="white")
        else:
            renderable = Text(str(detail.get("body", "")), style="white")

        if isinstance(renderable, Panel):
            self._ui_print(renderable)
        else:
            self._ui_print(Panel(renderable, title=title, border_style=border_style, expand=True))

    def _render_assistant_message(self, assistant_msg: ChatMessage) -> None:
        content = (assistant_msg.content or "").strip()
        if not content:
            return
        if assistant_msg.tool_calls:
            self._render_event(content, label="reason", style="bright_black")
            return
        if self.verbosity >= 2:
            self._render_detail(
                {
                    "kind": "text",
                    "title": f"Assistant [{self.name}]",
                    "body": content,
                    "border_style": "cyan",
                },
                force=True,
            )
        else:
            self._render_event(_clip(content, 120), style="white")

    def _render_usage(self, assistant_msg: ChatMessage) -> None:
        if not getattr(assistant_msg, "usage", None):
            return
        prompt_tokens = getattr(assistant_msg.usage, "prompt_tokens", None)
        if prompt_tokens is not None:
            try:
                self._latest_prompt_tokens = int(prompt_tokens)
            except Exception:
                self._latest_prompt_tokens = None
            self._refresh_live()
        if self.verbosity < 2:
            return
        total_tokens = getattr(assistant_msg.usage, "total_tokens", None)
        if prompt_tokens is not None:
            self._render_event(f"Prompt tokens: {prompt_tokens}", level=2, label="", style="bright_black")
        if total_tokens is not None:
            self._render_event(f"Total tokens: {total_tokens}", level=2, label="", style="bright_black")

    def _prompt_for_approval(self, tool: Tool, args: dict[str, Any], preview: Mapping[str, Any] | None) -> tuple[bool, str | None]:
        if preview:
            self._render_detail(preview, force=True)
        prompt = f"Approve {tool.name}: {tool.describe_call(**args)}? [y/N/m or feedback] "
        while True:
            response = self._ui_input(prompt).strip()
            lowered = response.lower()
            if lowered == "y":
                return True, None
            if lowered == "m":
                self._render_detail(
                    {
                        "kind": "json",
                        "title": f"Arguments for {tool.name}",
                        "data": args,
                        "border_style": "magenta",
                    },
                    force=True,
                )
                continue
            if lowered in {"", "n"}:
                feedback = self._ui_input("Feedback (optional): ").strip()
                return False, feedback or None
            return False, response

    def _append_to_summary(self, summary: str) -> None:  # noqa: D401
        os.makedirs(self._trace_dir, exist_ok=True)
        with open(os.path.join(self._trace_dir, "summary.txt"), "a", encoding="utf-8") as f:
            f.write(f"[{self._global_step}] {self.name} > {summary}\n")

    @staticmethod
    def _trace_content(content: Any) -> str:
        if not isinstance(content, list):
            return str(content or "")
        lines = []
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type", "unknown")
            if part_type == "text":
                lines.append(f"[Text content]: {part.get('text', '')}")
                continue
            if part_type == "image_url":
                image_url = part.get("image_url", {}).get("url", "")
                label = f"<image data - {len(image_url):,} characters>" if image_url.startswith("data:image") else image_url
                lines.append(f"[Image content]: {label}")
                continue
            lines.append(f"[Other content type: {part_type}]")
        return "\n".join(lines)

    @staticmethod
    def _trace_tool_calls(tool_calls: Any) -> str:
        serialised = []
        for tc in tool_calls or []:
            args = tc["function"].get("arguments", {})
            try:
                args = json.loads(args) if isinstance(args, str) else args
            except Exception:
                args = str(args)
            if isinstance(args, dict):
                args = {
                    key: textwrap.fill(value, width=80) if isinstance(value, str) and len(value) > 100 else value
                    for key, value in args.items()
                }
            serialised.append(
                {
                    "id": tc.get("id", ""),
                    "type": "function",
                    "function": {"name": tc["function"]["name"], "arguments": args},
                }
            )
        return json.dumps(serialised, indent=2, ensure_ascii=False)

    def _dump_trace(self) -> None:  # noqa: D401
        os.makedirs(self._trace_dir, exist_ok=True)
        fname = os.path.join(self._trace_dir, f"step_{self._global_step:03d}_{self.name}.log")
        blocks = []
        for idx, msg in enumerate(self.memory, 1):
            if msg.role == "assistant":
                lines = [f"[{idx}] ASSISTANT [{self.name}]:", str(msg.content or "")]
                if msg.tool_calls:
                    lines.extend(["TOOL CALLS:", self._trace_tool_calls(msg.tool_calls)])
                if getattr(msg, "usage", None):
                    lines.append(f"[TOKENS: {getattr(msg.usage, 'total_tokens', 'unknown')}]")
                blocks.append("\n".join(lines))
                continue
            label = f"[{idx}] TOOL RESPONSE from {msg.name}:" if msg.role == "tool" else f"[{idx}] {msg.role.upper()}:"
            blocks.append(f"{label}\n{self._trace_content(msg.content)}")
        with open(fname, "w", encoding="utf-8") as f:
            f.write(("\n" + "-" * 80 + "\n\n").join(blocks))
        self._log(f"Trace dumped to {fname}", 3)

    def _build_system_prompt(self, base: str, add_tools: bool) -> str:  # noqa: D401
        prompt = base.strip() + "\n\n" + textwrap.dedent(
            """
            Tool-use guidance:
            - Before a meaningful cluster of tool calls, use the commentary tool to briefly explain what you are about to do.
            - Use commentary to connect progress across steps, especially after learning something important, changing approach, or beginning implementation/verification.
            - Keep commentary to 1-2 sentences, grounded in recent progress, and focused on the immediate next actions.
            - Skip commentary for isolated trivial reads.
            - Avoid long silent stretches of tool calls; after several meaningful actions, add another concise commentary update.
            - Do not use commentary or update_plan as a substitute for making progress; after narrating or planning, do the work.
            """
        ).strip()
        if not add_tools:
            return prompt
        managed = list(self.managed_agents.values())
        lines = [prompt, "", "Here are your tools:"]
        lines.extend(
            f"- {tool.name}: (inputs: {', '.join(f'{k}: {v['type']}' for k, v in tool.inputs.items()) or 'None'})"
            for tool in self.tools.values()
            if tool not in managed
        )
        if self.managed_agents:
            lines.extend(
                [
                    "\nYou can also give tasks to team members:",
                    "Calling a team member works the same as calling a tool: the only argument you need to provide is 'task', a string explaining what you want them to do.",
                    "Since these team members are specialized agents, be clear and detailed in your task descriptions.",
                ]
            )
            for agent in self.managed_agents.values():
                description = getattr(agent, "description", "")
                if getattr(agent, "clear_memory_on_run", False):
                    description += " This team member starts with a clean memory for each task and needs comprehensive context."
                lines.append(f"- {agent.name}: {description}")
        return "\n".join(lines)

    def _recover_tool_calls(self, response_content: str | None, tool_calls: Any) -> tuple[str | None, Any]:
        if tool_calls is not None:
            return response_content, tool_calls
        source = response_content or ""
        if self.include_function_thoughts:
            thought_part, tool_call_part = self._split_response_with_thoughts(source)
            maybe_calls = self._extract_tool_calls(tool_call_part) if thought_part and tool_call_part else None
            if maybe_calls:
                return thought_part, maybe_calls
        maybe_calls = self._extract_tool_calls(source)
        return (None if maybe_calls else response_content), maybe_calls

    def _tool_preview(self, name: str, target: Tool, args: dict[str, Any]) -> tuple[bool, Mapping[str, Any] | None]:
        try:
            requires_confirmation = target.needs_confirmation(**args)
        except Exception:
            requires_confirmation = False
        if not requires_confirmation and self.verbosity < 2:
            return requires_confirmation, None
        try:
            return requires_confirmation, target.build_preview(**args)
        except Exception as exc:
            return requires_confirmation, {
                "kind": "text",
                "title": f"Preview unavailable for {name}",
                "body": str(exc),
                "border_style": "red",
            }

    def _run_tool(self, name: str, target: Tool, args: dict[str, Any]) -> tuple[Mapping[str, Any] | None, bool, Any, int]:
        requires_confirmation, preview = self._tool_preview(name, target, args)
        approved, rejection_feedback, preview_shown = True, None, False
        if requires_confirmation:
            self._status_message = f"Awaiting approval: {name}"
            self._refresh_live()
            approved, rejection_feedback = self._prompt_for_approval(target, args, preview)
            preview_shown = preview is not None
        try:
            if approved:
                tool_started = time.perf_counter()
                return preview, preview_shown, target(**args), int((time.perf_counter() - tool_started) * 1000)
            return preview, preview_shown, target.rejection_result(rejection_feedback, **args), 0
        except Exception as exc:  # pragma: no cover - runtime errors
            return preview, preview_shown, f"ToolError[{name}]: {exc} ({traceback.format_exc().splitlines()[-1]})", 0

    @staticmethod
    def _image_part(url: str) -> dict[str, Any]:
        return {"type": "image_url", "image_url": {"url": url, "detail": "auto"}}

    @staticmethod
    def _shell_result_for_memory(result: Any) -> str:
        parsed = _parse_command_result(str(result))
        if parsed.get("exit_code") is None:
            return str(result)
        lines = [f"[exit code] {parsed['exit_code']}"]
        if parsed.get("status"):
            lines.append(f"[status] {parsed['status']}")
        output = parsed.get("output")
        if output:
            lines.append(output)
        return "\n".join(lines)

    def _content_for_memory(
        self,
        target: Tool,
        result: Any,
        args: dict[str, Any],
        image_parts: list[dict[str, Any]],
    ) -> str:
        if getattr(target, "name", "") == "shell":
            return self._shell_result_for_memory(result)
        if getattr(target, "output_type", "") == "image" and isinstance(result, str) and result.startswith("data:image"):
            image_parts.append(self._image_part(result))
            return f"<{os.path.basename(args.get('filename', 'image'))} • {len(result):,} chars>"
        if getattr(target, "output_type", "") == "object" and isinstance(result, dict) and "images" in result:
            text_blocks = [
                f"Page {item['page']}:\n{item['content'].strip()}"
                for item in result.get("text", [])
                if "page" in item and item.get("content", "").strip()
            ]
            if text_blocks:
                image_parts.append({"type": "text", "text": "Extracted PDF Text:\n\n" + "\n\n".join(text_blocks)})
            image_parts.extend(
                self._image_part(img["data"])
                for img in result.get("images", [])
                if img.get("data", "").startswith("data:image")
            )
            page_info = f"page {args.get('page', 'all')}" if "page" in args else "all pages"
            return (
                f"<PDF: {os.path.basename(args.get('filename', 'document.pdf'))} • {page_info} • "
                f"{len(result.get('text', []))} text blocks, {len(result.get('images', []))} images>"
            )
        return str(result)

    def _remember_plan(self, target: Tool) -> None:
        last_plan = getattr(target, "_last_plan", None)
        if not last_plan:
            return
        self._current_plan = {
            "explanation": last_plan.get("explanation"),
            "plan": list(last_plan.get("plan", [])),
        }
        self._refresh_live()

    @staticmethod
    def _result_style(result_summary: str) -> str:
        lowered = result_summary.lower()
        if "rejected by user" in lowered:
            return "yellow"
        if any(term in lowered for term in ("toolerror", "failed", "not found", "invalid", "timed out")) or lowered.startswith("error"):
            return "red"
        return "green"

    def _result_summary(self, target: Tool, result: Any, args: dict[str, Any], elapsed_ms: int) -> str:
        try:
            summary = target.describe_result(result, **args)
        except Exception:
            lines = str(result).splitlines()
            summary = _clip(lines[0] if lines else str(result), 120)
        return f"{summary} ({elapsed_ms} ms)" if elapsed_ms and summary else summary

    def _render_tool_details(self, name: str, target: Tool, result: Any, args: dict[str, Any]) -> None:
        try:
            result_detail = target.build_result_details(result, **args)
        except Exception as exc:
            result_detail = {
                "kind": "text",
                "title": f"{name} details unavailable",
                "body": str(exc),
                "border_style": "red",
            }
        self._render_detail(result_detail, force=bool(name == "update_plan" and result_detail and self.verbosity >= 1))

    def _take_action(self, specific_tools=None) -> str | None:  # noqa: D401
        tools = [self._tool_to_openai(t) for t in self.tools.values() if specific_tools is None or t.name in specific_tools]
        self._status_message = "Thinking"
        self._refresh_live()
        msg = self.model.chat(messages=self.memory.to_openai(), tools=tools)
        response_content, tool_calls = self._recover_tool_calls(
            msg.content,
            _normalise_tool_calls(getattr(msg, "tool_calls", None)),
        )
        assistant_msg = ChatMessage.assistant(response_content, tool_calls=tool_calls)
        if hasattr(msg, "usage"):
            assistant_msg.usage = msg.usage
        self.memory.append(assistant_msg)
        self._render_assistant_message(assistant_msg)
        self._render_usage(assistant_msg)
        if not tool_calls:
            self._log("No tool call used, instructing the agent to try again", 1)
            self.memory.append(ChatMessage.user(
                "You must use a tool call. Think about which tool will let you proceed. Use final_answer tool if you are FINISHED, otherwise use a different tool."
            ))
            return None
        if self.trace_enabled:
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
                self._render_event(err, label="", style="red")
                continue
            try:
                call_summary = target.describe_call(**args)
            except Exception:
                call_summary = name
            call_display = f"{name}: {call_summary}"
            self._append_to_summary(call_display)
            self._status_message = call_display
            self._refresh_live()
            compact_ui = self._compact_single_agent_ui and not self.managed_agents
            self._render_event(
                call_summary if (name == "commentary" or compact_ui) else call_display,
                label="commentary" if name == "commentary" else (name if compact_ui else None),
                style="white" if name == "commentary" else "cyan",
            )
            preview, preview_shown, result, elapsed_ms = self._run_tool(name, target, args)
            self.memory.append(
                ChatMessage.tool(
                    name=name,
                    tool_call_id=tc.get("id", "call_0"),
                    result=self._content_for_memory(target, result, args, image_parts),
                )
            )
            if name == "final_answer":
                final_answer = args.get("answer", str(result))
            elif name == "update_plan":
                self._remember_plan(target)
            result_summary = self._result_summary(target, result, args, elapsed_ms)
            if result_summary:
                self._render_event(result_summary, label="", style=self._result_style(result_summary))
            if preview and not preview_shown and self.verbosity >= 2 and preview.get("kind") in {"diff", "code"}:
                self._render_detail(preview)
            self._render_tool_details(name, target, result, args)
            if name != "commentary":
                self._status_message = result_summary or f"{name} complete"
                self._refresh_live()
        if image_parts:
            self.memory.append(ChatMessage.user(image_parts))
        return final_answer

    def _summarize_for_final(self) -> str:  # noqa: D401
        summary_prompt = "Summarise the current progress so the user can continue on their own."
        msg = self.model.chat(
            messages=self.memory.to_openai() + [ChatMessage.user(summary_prompt).to_openai()],
            tools=[self._tool_to_openai(self.tools["final_answer"])],
        )
        tool_calls = _normalise_tool_calls(getattr(msg, "tool_calls", None))
        if not tool_calls:
            return msg.content
        tc = tool_calls[0]
        if tc and tc["function"]["name"] == "final_answer":
            return json.loads(tc["function"].get("arguments", "{}")).get("answer", "")
        return msg.content or ""

    # ------------------------------------------------------------------
    @staticmethod
    def _tool_to_openai(tool: Tool) -> dict[str, Any]:  # noqa: D401
        properties: dict[str, Any] = {}
        required: list[str] = []
        for k, v in tool.inputs.items():
            prop = {pk: pv for pk, pv in v.items() if pk != "required"}
            if prop.get("type") == "any":
                # OpenAI tool schemas do not accept a literal JSON Schema type "any".
                # Omitting the type leaves the property unconstrained.
                prop.pop("type", None)
            elif prop.get("type") == "array" and "items" not in prop:
                # OpenAI requires array schemas to specify an items schema.
                prop["items"] = {}
            elif prop.get("type") == "object" and "properties" not in prop and "additionalProperties" not in prop:
                # Allow free-form objects for tools like shell.env.
                prop["additionalProperties"] = True
            properties[k] = prop
            if v.get("required", True):
                required.append(k)

        parameters: dict[str, Any] = {
            "type": "object",
            "properties": properties,
        }
        if required:
            parameters["required"] = required

        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": parameters,
            },
        }

    @staticmethod
    def _split_response_with_thoughts(text: str) -> tuple[str | None, str | None]:
        if not text:
            return None, None
        matches = [match for pat in _PATTERNS if (match := pat.search(text))]
        if not matches:
            return None, None
        tool_call_start = min(match.start() for match in matches)
        thought_part = text[:tool_call_start].strip() or None
        tool_call_part = text[tool_call_start:].strip() or None
        return thought_part, tool_call_part

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
                    if pat in (_PAT_TOOL_CALLS_SPECIAL, _PAT_TOOL_CALLS_DIRECT, _PAT_GPT_OSS):
                        data = {"name": m.group(1), "arguments": json.loads(m.group(2))}
                    elif pat in (_PAT_FUNC, _PAT_TOOL_CALLS_ARGS):
                        data = {"name": m.group(1), "arguments": _json_from_blob(m.group(2))}
                    else:
                        data = _json_from_blob(m.group(1))
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

    def snapshot(self) -> dict:  # noqa: D401
        return {
            "version": 1,
            "timestamp": f"{_dt.datetime.now():%Y-%m-%dT%H:%M:%S}",
            "agent": {
                "name": self.name,
                "description": self.description,
                "max_steps": self.max_steps,
                "verbosity": self.verbosity,
                "trace_enabled": self.trace_enabled,
                "planning_interval": self.planning_interval,
                "memory_threshold": self.memory_threshold,
                "include_function_thoughts": self.include_function_thoughts,
            },
            "model": {
                "model_id": getattr(self.model, "model_id", None),
                "temperature": getattr(self.model, "temperature", None),
                "debug": getattr(self.model, "debug", None),
            },
            "memory": [m.to_session() for m in self.memory],
            "global_step": self._global_step,
            "cwd": os.getcwd(),
            "trace_dir": getattr(self, "_trace_dir", None),
            "current_plan": self._current_plan,
        }

    def save_session(self, path = None) -> None:  # noqa: D401
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        else:
            session_dir = os.path.join(os.getcwd(), ".agent_sessions")
            os.makedirs(session_dir, exist_ok=True)
            path = os.path.join(session_dir, f"{self.name}_{_dt.datetime.now():%y%m%d_%H%M%S}.json")

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.snapshot(), f, indent=2)
        _console_print(f"Session saved to {path}", style="bright_black")
        _console_print(f"Resume with:\n  --resume \"{path}\"", style="bright_black")

    @staticmethod
    def load_session(path: str) -> dict:  # noqa: D401
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @classmethod
    def from_session(cls, session: dict, *, model: "LLMClient", tools: Sequence["Tool"]) -> "Agent":  # noqa: D401
        ag = session.get("agent", {})
        inst = cls(
            tools=tools,
            model=model,
            name=ag.get("name", "agent"),
            description=ag.get("description", ""),
            max_steps=ag.get("max_steps", 20),
            verbosity=ag.get("verbosity", 2),
            trace_enabled=ag.get("trace_enabled", True),
            planning_interval=ag.get("planning_interval"),
            memory_threshold=ag.get("memory_threshold"),
            include_function_thoughts=ag.get("include_function_thoughts", True),
        )
        inst.memory = ConversationMemory()
        for msg in session.get("memory", []):
            inst.memory.append(ChatMessage.from_session(msg))
        try:
            cls._global_step = int(session.get("global_step", 0))
        except Exception:
            pass
        td = session.get("trace_dir")
        if td:
            inst._trace_dir = td
        inst._current_plan = session.get("current_plan")
        return inst

def _find_latest_session(session_dir: str) -> Optional[str]:  # noqa: D401
    try:
        if not os.path.isdir(session_dir):
            return None
        files = [
            os.path.join(session_dir, f)
            for f in os.listdir(session_dir)
            if f.endswith(".json") and os.path.isfile(os.path.join(session_dir, f))
        ]
        if not files:
            return None
        files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return files[0]
    except Exception:
        return None

_MANAGER_PROMPT = textwrap.dedent('''You are the *manager_agent* - a senior engineer. Your first
    task is to use the update_plan tool. In the plan you should start with
    Goal: <the overall goal of the task -- restating it in your own words>
    Completion Criteria: <what did the user specify that would complete the task. Be complete and precise>
    Next, break the user's high-level request into a numbered sequence of
    concrete, executable steps (and sub-steps if necessary). Each step MUST include:
    (1) a clear instruction for the code_agent
    (2) any parameters or file names needed, and
    (3) explicit completion criteria. You are the expert in this area,
    so be very clear about the details and instructions so the code_agent doesn't
    have to fill in too many blanks. If code is requested, make sure to ask for it to
    be run and tested also.

    Your plan should not be overly complex -- accomplish the task in the minimal
    number of steps necessary. Do NOT ask the agent to write scaffolding files first.
    Do NOT ask the agent not use a virtual environment or git repo or install anyything.

    Then you must delegate the to the `code_agent` and
    wait for its report. You should delegate enough steps that the code_agent can complete
    then in a ~few function calls (eg write a few files and test them).
    This agent does not know ANY context about the task, so
    be sure to provide complete instructions and background information.

    When a step is reported complete, mark it as
    done and delegate the next. The code_agent does not know of any
    previous work so be sure to give complete and verbose context. At each
    step you must give full context to the code_agent, including a summary of any
    previous steps.

    Once you are confident all steps have been FULLY completed, pass the results
    to the judge agent, fully explaining what the task was, the completion criteria, and
    the solution. The judge agent will look at the results and report back if the
    task is finished or not. Only after this has been complete, you can call final_answer
    and report the results to the user.''')

_JUDGE_PROMPT = (
    "You are the *judge_agent* - an impartial evaluator. Your role is to objectively assess work against "
    "specific success criteria. When evaluating, you should:\n"
    "1. Clearly state each success criterion\n"
    "2. Examine the relevant files/outputs to determine if criteria are met\n"
    "3. For each criterion, provide a clear PASS/FAIL assessment with evidence\n"
    "4. For any failures, explain why the criterion wasn't met\n"
    "Your evaluation should be thorough, evidence-based, and focus solely on whether the specified "
    "success criteria have been met, not on subjective qualities of the implementation."
)


def _read_task(task: str | None) -> str | None:
    if not task:
        return None
    if not task.endswith(".txt"):
        return task
    with open(task, encoding="utf-8") as f:
        return f.read()


def _resolve_resume_path(args: argparse.Namespace) -> str | None:
    if not args.resume:
        return None
    if args.resume != "LATEST":
        return args.resume
    session_dir = args.session_dir or os.path.join(os.getcwd(), ".ants_sessions")
    resume_path = _find_latest_session(session_dir)
    message = f"Resuming latest session: {resume_path}" if resume_path else f"No session found in {session_dir}; starting fresh."
    _console_print(message, style="bright_black")
    return resume_path


def _copy_into_workdir(source_path: str, workdir: str) -> None:
    if not source_path:
        return
    if not os.path.exists(source_path):
        _console_print(f"Warning: Source path {source_path} not found, skipping copy", style="yellow")
        return
    dest_path = os.path.join(workdir, os.path.basename(source_path))
    _console_print(
        f"copying {'directory' if os.path.isdir(source_path) else 'file'} {source_path} to {dest_path}",
        style="bright_black",
    )
    if os.path.isdir(source_path):
        shutil.copytree(source_path, dest_path)
    else:
        shutil.copy2(source_path, dest_path)


def _enter_workdir(args: argparse.Namespace) -> str | None:
    if not args.wkdir:
        return None
    cwd = os.getcwd()
    workdir = os.path.join(cwd, f"work/tmp_{_dt.datetime.now():%y%m%d_%H%M}")
    os.makedirs(workdir, exist_ok=True)
    _console_print(f"moving to {workdir}", style="bright_black")
    _copy_into_workdir(args.cp, workdir)
    os.chdir(workdir)
    return cwd


def _code_tools(args: argparse.Namespace) -> list[Tool]:
    tools = [
        WriteFile(confirm_edits=args.confirm_edits),
        ReadFile(),
        SearchFiles(),
        EditFile(confirm_edits=args.confirm_edits),
        Shell(confirm_commands=args.confirm_shell),
        Commentary(),
        UpdatePlan(),
        ListFiles(),
        FinalAnswer(),
    ]
    if not args.multi:
        tools.insert(-1, ReadPDF())
    if args.vision:
        tools.insert(-1, ViewImage())
    return tools


def _manager_tools(args: argparse.Namespace) -> list[Tool]:
    tools = [
        ListFiles(),
        SearchFiles(),
        MakePlan(),
        Reflect(),
        Commentary(),
        ReadFile(),
        RunPython(),
        Shell(confirm_commands=args.confirm_shell),
        FinalAnswer(),
    ]
    if args.vision:
        tools[-1:-1] = [ReadPDF(), ViewImage()]
    if args.confirm_plan:
        tools.append(GetUserInput())
    return tools


def _build_code_agent(args: argparse.Namespace, model: LLMClient, tools: list[Tool], resume_path: str | None) -> Agent:
    if not resume_path:
        return Agent(
            tools=tools,
            model=model,
            max_steps=25,
            verbosity=args.verbosity,
            trace_enabled=not args.no_trace,
            planning_interval=args.planning,
            name="code_agent",
            description="Writes/tests Python projects",
        )
    session = Agent.load_session(resume_path)
    if args.chdir_on_resume and "cwd" in session and os.path.isdir(session["cwd"]):
        os.chdir(session["cwd"])
    agent = Agent.from_session(session, model=model, tools=tools)
    agent.trace_enabled = not args.no_trace
    return agent


def _build_multi_agent(args: argparse.Namespace, model: LLMClient, agent_code: Agent) -> Agent:
    agent_code.clear_memory_on_run = True
    tools = _manager_tools(args)
    judge_agent = Agent(
        tools=tools,
        model=model,
        system_message=_JUDGE_PROMPT,
        verbosity=args.verbosity,
        trace_enabled=not args.no_trace,
        max_steps=15,
        name="judge_agent",
        description="Evaluates work against success criteria and provides objective assessment.",
        clear_memory_on_run=True,
    )
    return Agent(
        tools=tools,
        model=model,
        system_message=_MANAGER_PROMPT,
        managed_agents=[agent_code, judge_agent],
        max_steps=25,
        verbosity=args.verbosity,
        trace_enabled=not args.no_trace,
        planning_interval=args.planning,
        name="manager_agent",
        description="Magnages coding agents",
    )


def _interactive_loop(agent: Agent) -> None:
    while True:
        _console_print("end to quit • compact to summarize memory", style="italic bright_black")
        follow = _console_input("› ").strip()
        cmd = follow.lower()
        if cmd == "end":
            return
        if cmd == "compact":
            before = len(agent.memory)
            agent.memory.summarize(agent.model)
            _console_print(f"Memory compacted: {before} -> {len(agent.memory)} messages", style="bright_black")
            agent.save_session()
            continue
        agent.max_steps += 20
        _render_final_answer_panel(agent.run(follow, reset=False), title="Assistant")
        agent.save_session()


def main() -> None:  # noqa: D401
    parser = argparse.ArgumentParser(description="Run the autonomous Agent")
    parser.add_argument("task", nargs="?", help="Initial user task (prompted if omitted)")
    parser.add_argument("-d", "--debug", action="store_true", help="Verbose OpenAI request/response logging")
    parser.add_argument("-l", "--local", action="store_true",
                        help="Use a local LLM instead of the OpenAI API for the executor agent")
    parser.add_argument("-v", "--verbosity", type=int, default=2, choices=range(0,4),
                        help="Verbosity level: 0-quiet 1-timeline 2-details 3-verbose (default 2)")
    parser.add_argument("-m", "--model", default="gpt-5.4", help="OpenAI model")
    parser.add_argument("-w", "--wkdir", action="store_true", help="move to work dir")
    parser.add_argument("--cp", default="", help="Copy file or directory to wkdir")
    parser.add_argument("-c", "--confirm-edits", action="store_true", 
                        help="Require confirmation before writing, editing, or deleting files")
    parser.add_argument("--confirm-shell", action="store_true",
                        help="Require confirmation before running shell commands")
    parser.add_argument("-p", "--planning",default=0,type=int,
                        help="Planning interval")
    parser.add_argument("--confirm-plan", action="store_true", 
                        help="Ask for confirmation after making initial plan")
    parser.add_argument("--no-trace", action="store_true",
                        help="Disable per-step trace file dumps")
    parser.add_argument( "--end", action="store_true", 
                        help="End on first final_answer")
    parser.add_argument("--multi", action="store_true",
                        help="Use multiple agents starting with a manager")
    parser.add_argument( "--vision", action="store_true", 
                        help="The model has vision and can use view_image tool")
    parser.add_argument("--resume", nargs="?", const="LATEST", help="Resume from a session JSON (omit value to resume latest)")
    parser.add_argument("--session-dir", default=None, help="Directory to save sessions on 'end' (default ./.ants_sessions)")
    parser.add_argument("--chdir-on-resume", action="store_true", help="chdir to session cwd on resume")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger("agent").setLevel(logging.INFO)
        logging.getLogger("agent.tools").setLevel(logging.INFO)
    resume_path = _resolve_resume_path(args)
    user_task = _read_task(args.task) if args.task or resume_path else _console_input("Enter your task: ")
    if resume_path and not args.task:
        user_task = None
    cwd = _enter_workdir(args)
    tools_all = _code_tools(args)

    model = LLMClient(
        model_id="lmstudio" if args.local else args.model,
        debug=args.debug,
        api_base="http://localhost:1234/v1" if args.local else None,
    )
    agent = _build_code_agent(args, model, tools_all, resume_path)
    if args.multi:
        agent = _build_multi_agent(args, model, agent)
    if user_task:
        result = agent.run(user_task, reset=False if resume_path else None)
        _render_final_answer_panel(result)
        agent.save_session()
    if not args.end:
        _interactive_loop(agent)
    if cwd:
        os.chdir(cwd)

if __name__ == "__main__":  # pragma: no cover
    main()
