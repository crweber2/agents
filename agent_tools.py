"""Agent Tools Module.

This module provides a collection of tool classes used by the Agent class to perform
various operations on files, execute code, and interact with the system. These tools
are designed to be used with the Agent class from the agents.py module.

Key components:
- Tool: Base class for all tools
- WriteFile: Create or overwrite files
- ReadFile: Read file contents
- EditFile: Search and replace text in files
- RunPython: Execute Python scripts
- RunBash: Run shell commands
- Delete: Remove files or directories
- ListFiles: List directory contents
- ViewImage: Display images to the agent
- ReadPDF: Read PDF files with full page images for complex content
- MakePlan: Create step-by-step plans
- UpdatePlan: Update step-by-step plans
- FinalAnswer: Return final answers to the user
- GetUserInput: Request information from the user

Usage:
    from agent_tools import WriteFile, ReadFile, RunPython
    
    # Create tool instances
    write_tool = WriteFile()
    read_tool = ReadFile()
    
    # Use tools directly
    write_tool(filename="example.py", code="print('Hello, world!')")
    content = read_tool(filename="example.py")

Author: Chris Weber, crweber@gmail.com
"""

from __future__ import annotations

###############################################################################
# Imports
###############################################################################

# stdlib ---------------------------------------------------------------------
import base64
import difflib
import json
import logging
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import textwrap
import traceback
from typing import Any, Dict, Mapping, Optional, Union

###############################################################################
# Helpers & constants
###############################################################################

LOGGER = logging.getLogger("agent.tools")

def truncate(s: str, max_length: int = 20000) -> str:
    """Truncate a string to a maximum length with clear indication of truncation."""
    if len(s) <= max_length:
        return s
    
    # Keep half from start and half from end
    half = max_length // 2
    truncation_msg = f'\n<... {len(s) - max_length} characters truncated ...>\n'
    
    return s[:half] + truncation_msg + s[-half:]

authorized_types = {
    "string",
    "boolean",
    "integer",
    "number",
    "image",
    "audio",
    "array",
    "object",
    "any",
    "null",
}

_RE_TRAILING_COMMA = re.compile(r",\s*([}\]])")


def _clip(text: str, max_length: int = 90) -> str:
    text = str(text).strip().replace("\n", " ")
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def _line_count(text: str) -> int:
    if not text:
        return 0
    return text.count("\n") + 1


def _safe_read_text(path: str) -> str | None:
    try:
        with open(path, "r", encoding="utf-8") as fp:
            return fp.read()
    except (FileNotFoundError, IsADirectoryError, UnicodeDecodeError, OSError):
        return None


def _build_unified_diff(original: str, updated: str, *, path: str, proposed_label: str | None = None) -> str:
    proposed = proposed_label or f"{path} (proposed)"
    diff = difflib.unified_diff(
        original.splitlines(),
        updated.splitlines(),
        fromfile=path,
        tofile=proposed,
        lineterm="",
    )
    return "\n".join(diff)


def _stringify_command(command: Union[str, list[str]]) -> str:
    if isinstance(command, str):
        return command
    return " ".join(command)


def _render_command_result(
    *,
    command_display: str,
    cwd: str,
    returncode: int,
    output: str,
    max_chars: int,
    notes: Optional[list[str]] = None,
    timed_out: bool = False,
) -> str:
    header = []
    if notes:
        header.extend(notes)
    header.extend(
        [
            f"[cwd] {os.path.abspath(cwd)}",
            f"[command] {command_display}",
            f"[exit code] {returncode}",
        ]
    )
    if timed_out:
        header.append("[status] command timed out")
    elif returncode != 0:
        header.append("[status] command failed (non-zero exit)")
    rendered = "\n".join(header)
    if output:
        rendered += "\n" + output
    return truncate(rendered, max_length=max_chars)


def _parse_command_result(result: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {
        "notes": [],
        "cwd": None,
        "command": None,
        "exit_code": None,
        "status": None,
        "output": "",
    }
    lines = str(result).splitlines()
    body_start = 0
    for idx, line in enumerate(lines):
        if line.startswith("[cwd] "):
            parsed["cwd"] = line[6:]
        elif line.startswith("[command] "):
            parsed["command"] = line[10:]
        elif line.startswith("[exit code] "):
            try:
                parsed["exit_code"] = int(line[12:])
            except ValueError:
                parsed["exit_code"] = None
        elif line.startswith("[status] "):
            parsed["status"] = line[9:]
        elif line.startswith("[note] ") or line.startswith("[justification] "):
            parsed["notes"].append(line)
        else:
            body_start = idx
            break
    else:
        body_start = len(lines)
    parsed["output"] = "\n".join(lines[body_start:]).strip()
    return parsed


def _tail(text: str, max_lines: int = 40) -> str:
    lines = str(text).splitlines()
    if len(lines) <= max_lines:
        return "\n".join(lines)
    omitted = len(lines) - max_lines
    return f"... {omitted} line(s) omitted ...\n" + "\n".join(lines[-max_lines:])

###############################################################################
# Tool base class and built-in tools
###############################################################################


class Tool:
    """Minimal *smolagent*-style tool interface."""

    name: str
    description: str
    inputs: Mapping[str, Mapping[str, str]]
    output_type: str

    def __init__(self, **config) -> None:
        self.config = config
        self._validate()

    # ------------------------------------------------------------------
    def _validate(self) -> None:  # noqa: D401
        for attr in ("name", "description", "inputs", "output_type"):
            if getattr(self.__class__, attr, None) is None:  # pragma: no cover - dev error guard
                raise TypeError(f"Tool class missing required attribute {attr}")
        if self.output_type not in authorized_types:
            raise ValueError(f"{self.name}: invalid output_type '{self.output_type}'.")

    # ------------------------------------------------------------------
    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # noqa: D401
        # Allow dict-style call for convenience
        if args and not kwargs and isinstance(args[0], dict):
            kwargs = args[0]
            args = ()
        return self.forward(*args, **kwargs)

    def forward(self, *_: Any, **__: Any) -> Any:  # noqa: D401
        raise NotImplementedError("Tools must implement forward()")

    def describe_call(self, **kwargs: Any) -> str:
        label = self.name.replace("_", " ")
        return label[:1].upper() + label[1:]

    def describe_result(self, result: Any, **kwargs: Any) -> str:
        if isinstance(result, dict):
            if "error" in result:
                return str(result["error"])
            keys = ", ".join(sorted(result.keys()))
            return f"Returned object ({keys})"
        if isinstance(result, list):
            return f"Returned {len(result)} item(s)"
        text = str(result).strip()
        if not text:
            return "Done"
        return _clip(text.splitlines()[0])

    def build_preview(self, **kwargs: Any) -> dict[str, Any] | None:
        return None

    def build_result_details(self, result: Any, **kwargs: Any) -> dict[str, Any] | None:
        return None

    def needs_confirmation(self, **kwargs: Any) -> bool:
        return False

    def rejection_result(self, feedback: str | None = None, **kwargs: Any) -> str:
        if feedback:
            return f"{self.describe_call(**kwargs)} rejected by user. Feedback: {feedback}"
        return f"{self.describe_call(**kwargs)} rejected by user."


# ――― Filesystem & execution tools ------------------------------------------


class WriteFile(Tool):
    name = "write_file"
    description = """Write text to disk (overwrites if exists).

    When to use:
    - Create a new file.
    - Overwrite an existing file when the changes are so extensive that
      multiple edit_file calls would be messy.

    Parameters:
    - filename [string] REQUIRED - target path (absolute or CWD‑relative).
    - code     [string] REQUIRED - full and final contents of the file.

    Usage example:
    {
      "tool": "write_file",
      "args": {
        "filename": "example.py",
        "code": "def hello_world():\\n    print('Hello, world!')\\n\\nif __name__ == '__main__':\\n    hello_world()\\n"
      }
    }

    Special notes:
    - Always supply the entire intended file, not a patch.
    - After the write the user's editor may auto‑format; read the final
      echo before drafting further edits.
    """
    inputs = {
        "filename": {"type": "string", "description": "Target path."},
        "code": {"type": "string", "description": "Content to write."},
    }
    output_type = "string"

    def describe_call(self, *, filename: str, **_: Any) -> str:
        return filename

    def describe_result(self, result: Any, *, filename: str, code: str, **_: Any) -> str:
        text = str(result)
        lowered = text.lower()
        if "rejected by user" in lowered or text.startswith("Error") or text.startswith("ToolError"):
            return text
        return f"Wrote {_line_count(code)} line(s) to {filename}"

    def build_preview(self, *, filename: str, code: str, **_: Any) -> dict[str, Any] | None:
        if os.path.isdir(filename):
            return {
                "kind": "text",
                "title": f"Cannot write {filename}",
                "body": f"{filename} is an existing directory.",
            }
        original = _safe_read_text(filename)
        if original is None:
            return {
                "kind": "code",
                "title": f"New file: {filename}",
                "body": code,
                "path": filename,
            }
        diff_text = _build_unified_diff(original, code, path=filename)
        if not diff_text:
            diff_text = f"No content changes for {filename}."
        return {
            "kind": "diff",
            "title": f"Proposed write: {filename}",
            "body": diff_text,
            "path": filename,
        }

    def needs_confirmation(self, **_: Any) -> bool:
        return bool(self.config.get("confirm_edits", False))

    def forward(self, *, filename: str, code: str) -> str:  # noqa: D401
        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
        with open(filename, "w", encoding="utf-8") as fp:
            fp.write(code)
        return f"Wrote {len(code)} bytes → {filename}"


class ReadFile(Tool):
    name = "read_file"
    description = """Return the entire UTF‑8 contents of a file.

    When to use:
    - You need to inspect or copy text not already visible in context.
    - You must capture exact lines for a forthcoming edit_file call.

    Parameters:
    - filename   [string]  REQUIRED - path to the file.
    - start_line [integer] OPTIONAL - 1-based first line to return (inclusive).
    - end_line   [integer] OPTIONAL - 1-based last line to return (inclusive).

    Usage example:
    {
      "tool": "read_file",
      "args": { "filename": "src/utils.py", "start_line": 20, "end_line": 60 }
    }

    Special notes:
    - Binary or very large files (>20 kB) are truncated with a notice.
    - If the user already pasted the file contents, reuse that copy
      instead of reading the file again.
    - If you've recently written the file or read it, you do NOT need to read_file again.
    - Only use start_line/end_line if you expect the file to be large, otherwise omit and 
      read the full file.
    """
    inputs = {
        "filename": {"type": "string", "description": "Path."},
        "start_line": {"type": "integer", "description": "Optional 1-based inclusive start line.", "required": False},
        "end_line": {"type": "integer", "description": "Optional 1-based inclusive end line.", "required": False},
    }
    output_type = "string"

    def describe_call(
        self,
        *,
        filename: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        **_: Any,
    ) -> str:
        if start_line is None and end_line is None:
            return filename
        if end_line is None:
            return f"{filename} from line {start_line}"
        return f"{filename} lines {start_line or 1}-{end_line}"

    def describe_result(self, result: Any, *, filename: str, **_: Any) -> str:
        text = str(result)
        if text.startswith("File not found") or text.startswith("Invalid") or text.startswith("Cannot decode"):
            return text
        if " lines " in text.splitlines()[0]:
            return _clip(text.splitlines()[0], 110)
        return f"Read {_line_count(text)} line(s) from {filename}"

    def forward(
        self,
        *,
        filename: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
    ) -> str:  # noqa: D401
        if not os.path.isfile(filename):
            return f"File not found: {filename}"
        if start_line is not None and start_line < 1:
            return "Invalid start_line: must be >= 1."
        if end_line is not None and end_line < 1:
            return "Invalid end_line: must be >= 1."
        if start_line is not None and end_line is not None and end_line < start_line:
            return "Invalid line range: end_line must be >= start_line."
        try:
            with open(filename, "r", encoding="utf-8") as fp:
                if start_line is None and end_line is None:
                    content = fp.read()
                    return truncate(content)

                lines = fp.readlines()
                total_lines = len(lines)
                if total_lines == 0:
                    return f"{filename} is empty."

                start_idx = 0 if start_line is None else min(start_line - 1, total_lines)
                end_idx = total_lines if end_line is None else min(end_line, total_lines)
                if start_idx >= total_lines:
                    return f"Requested start_line {start_line} is past EOF ({total_lines} lines)."

                width = max(4, len(str(end_idx)))
                selected = lines[start_idx:end_idx]
                numbered = "".join(
                    f"{i:>{width}}: {line}"
                    for i, line in enumerate(selected, start=start_idx + 1)
                )
                header = f"{filename} lines {start_idx + 1}-{start_idx + len(selected)} of {total_lines}\n"
                return truncate(header + numbered)
        except UnicodeDecodeError:
            return f"Cannot decode {filename} as UTF-8."


class EditFile(Tool):
    name = "edit_file"
    description = """Search & replace EXACT text segments inside a file.

    When to use:
    - Localised edits such as renaming a variable or tweaking a loop.
    - Batch several independent changes by stacking multiple calls.

    Parameters:
    - filename        [string] REQUIRED - file to modify.
    - search_string   [string] REQUIRED - text to find (must match whole lines).
    - replace_string  [string] REQUIRED - replacement text (empty ⇒ delete).

    Usage example:
    {
      "tool": "edit_file",
      "args": {
        "filename": "src/app.py",
        "search_string": "DEBUG = True",
        "replace_string": "DEBUG = False"
      }
    }

    Special notes:
    1. search_string must match whole lines exactly; partial matches fail.
    2. If no match is found the tool responds "No matches - nothing changed."
    3. For large refactors prefer write_file to avoid fragile search patterns.
    """
    inputs = {
        "filename": {"type": "string", "description": "Target file."},
        "search_string": {"type": "string", "description": "Text to search."},
        "replace_string": {"type": "string", "description": "Replacement text."},
    }
    output_type = "string"

    def describe_call(self, *, filename: str, **_: Any) -> str:
        return filename

    def describe_result(self, result: Any, *, filename: str, **_: Any) -> str:
        text = str(result)
        if text.startswith("Replaced "):
            return text.replace(" occurrence(s)", " change(s)")
        return text

    def build_preview(
        self,
        *,
        filename: str,
        search_string: str,
        replace_string: str,
        **_: Any,
    ) -> dict[str, Any] | None:
        original = _safe_read_text(filename)
        if original is None:
            return None
        patched = original.replace(search_string, replace_string)
        if original == patched:
            return {
                "kind": "text",
                "title": f"No proposed edits for {filename}",
                "body": "No matches - nothing changed.",
            }
        return {
            "kind": "diff",
            "title": f"Proposed edit: {filename}",
            "body": _build_unified_diff(original, patched, path=filename),
            "path": filename,
        }

    def needs_confirmation(self, **_: Any) -> bool:
        return bool(self.config.get("confirm_edits", False))

    def forward(self, *, filename: str, search_string: str, replace_string: str) -> str:  # noqa: D401
        if not os.path.isfile(filename):
            return f"File not found: {filename}"
        with open(filename, "r", encoding="utf-8") as fp:
            original = fp.read()
        patched = original.replace(search_string, replace_string)
        if original == patched:
            return "No matches - nothing changed."

        # Apply changes
        with open(filename, "w", encoding="utf-8") as fp:
            fp.write(patched)
        return f"Replaced {original.count(search_string)} occurrence(s) in {filename}"


APPLY_PATCH_TOOL_DESC = """This is a custom utility that makes it more convenient to add, remove, move, or edit code files. `apply_patch` effectively allows you to execute a diff/patch against a file, but the format of the diff specification is unique to this task, so pay careful attention to these instructions. To use the `apply_patch` command, you should pass a message of the following structure as "input":

%%bash
apply_patch <<"EOF"
*** Begin Patch
[YOUR_PATCH]
*** End Patch
EOF

Where [YOUR_PATCH] is the actual content of your patch, specified in the following V4A diff format.

*** [ACTION] File: [path/to/file] -> ACTION can be one of Add, Update, or Delete.
For each snippet of code that needs to be changed, repeat the following:
[context_before] -> See below for further instructions on context.
- [old_code] -> Precede the old code with a minus sign.
+ [new_code] -> Precede the new, replacement code with a plus sign.
[context_after] -> See below for further instructions on context.

For instructions on [context_before] and [context_after]:
- By default, show 3 lines of code immediately above and 3 lines immediately below each change. If a change is within 3 lines of a previous change, do NOT duplicate the first change’s [context_after] lines in the second change’s [context_before] lines.
- If 3 lines of context is insufficient to uniquely identify the snippet of code within the file, use the @@ operator to indicate the class or function to which the snippet belongs. For instance, we might have:
@@ class BaseClass
[3 lines of pre-context]
- [old_code]
+ [new_code]
[3 lines of post-context]

- If a code block is repeated so many times in a class or function such that even a single @@ statement and 3 lines of context cannot uniquely identify the snippet of code, you can use multiple `@@` statements to jump to the right context. For instance:

@@ class BaseClass
@@ 	def method():
[3 lines of pre-context]
- [old_code]
+ [new_code]
[3 lines of post-context]

Note, then, that we do not use line numbers in this diff format, as the context is enough to uniquely identify code. An example of a message that you might pass as "input" to this function, in order to apply a patch, is shown below.

%%bash
apply_patch <<"EOF"
*** Begin Patch
*** Update File: pygorithm/searching/binary_search.py
@@ class BaseClass
@@     def search():
-          pass
+          raise NotImplementedError()

@@ class Subclass
@@     def search():
-          pass
+          raise NotImplementedError()

*** End Patch
EOF
"""


class EditFile_patch(Tool):
    name = "apply_patch"
    description = APPLY_PATCH_TOOL_DESC #"Apply a context‑based diff to existing files. Uses the \"*** Begin Patch\" V4A format."
    inputs = {"patch": {"type": "string", "description": "Diff text beginning with *** Begin Patch"}} 
    output_type = "string"

    def describe_call(self, **_: Any) -> str:
        return "patch"

    def build_preview(self, *, patch: str, **_: Any) -> dict[str, Any] | None:
        return {
            "kind": "diff",
            "title": "Proposed patch",
            "body": patch,
        }

    def needs_confirmation(self, **_: Any) -> bool:
        return bool(self.config.get("confirm_edits", False))

    def forward(self, *, patch: str) -> str:
        import subprocess, tempfile, textwrap, os, sys
        with tempfile.NamedTemporaryFile("w", delete=False) as tf:
            tf.write(textwrap.dedent(patch))
            tf.flush()
            try:
                out = subprocess.check_output(["apply_patch", tf.name], stderr=subprocess.STDOUT, text=True)
                return out.strip()
            except subprocess.CalledProcessError as e:
                return f"Patch failed:\n{e.output}"
            finally:
                os.unlink(tf.name)

class RunPython(Tool):
    name = "run_python"
    description = """Execute a Python script in a subprocess and capture combined stdout/stderr.

    When to use:
    - Run an existing *.py* file end‑to‑end.
    - Surface runtime errors, printed output or visualisations.

    Parameters:
    - filename [string] REQUIRED - path to the script to execute.
    - args     [string] OPTIONAL - command-line arguments to pass to the script (e.g., "-v --count=5").
    - max_time [number] OPTIONAL - maximum execution time in seconds (default: 5 minutes).

    Usage example:
    {
      "tool": "run_python",
      "args": { 
        "filename": "demo.py",
        "args": "-v --input=data.csv",
        "max_time": 30
      }
    }

    Special notes:
    - Uses the same Python interpreter as the host process.
    - Output is captured and returned as a single result, truncated to the first 20 kB.
    - If execution exceeds max_time, the process is terminated with a timeout message.
    - Command-line arguments are passed as-is to the script.
    """
    inputs = {
        "filename": {"type": "string", "description": "Script path."},
        "args": {"type": "string", "description": "Command-line arguments to pass to the script (optional).", "required": False},
        "max_time": {"type": "number", "description": "Maximum execution time in seconds (optional).", "required": False}
    }
    output_type = "string"

    def describe_call(self, *, filename: str, args: str = "", **_: Any) -> str:
        suffix = f" {args}" if args else ""
        return f"{filename}{suffix}"

    def describe_result(self, result: Any, *, filename: str, **_: Any) -> str:
        parsed = _parse_command_result(str(result))
        exit_code = parsed.get("exit_code")
        if exit_code is None:
            return _clip(str(result).splitlines()[0])
        if exit_code == 0:
            return f"Python run succeeded for {filename}"
        return f"Python run failed for {filename} (exit {exit_code})"

    def build_result_details(self, result: Any, *, filename: str, **_: Any) -> dict[str, Any] | None:
        parsed = _parse_command_result(str(result))
        exit_code = parsed.get("exit_code")
        output = parsed.get("output") or ""
        if exit_code in (None, 0) and not output:
            return None
        if exit_code == 0:
            return None
        body = _tail(output)
        return {
            "kind": "text",
            "title": f"Python output: {filename}",
            "body": body or "(no output)",
        }

    def forward(self, *, filename: str, args: str = "", max_time: float = 300) -> str:  # noqa: D401
        if not os.path.isfile(filename):
            return f"File not found: {filename}"

        cmd = [sys.executable, filename]
        if args:
            import shlex
            cmd.extend(shlex.split(args))

        cwd = "."
        command_display = _stringify_command(cmd)
        max_chars = 20000
        try:
            result = subprocess.run(
                cmd,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=max_time,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            output = exc.output or ""
            output_str = output.decode("utf-8", errors="replace") if isinstance(output, bytes) else str(output)
            return _render_command_result(
                command_display=command_display,
                cwd=cwd,
                returncode=124,
                output=output_str,
                max_chars=max_chars,
                timed_out=True,
            )
        except Exception as exc:
            return f"Error running python: {exc}"

        return _render_command_result(
            command_display=command_display,
            cwd=cwd,
            returncode=result.returncode,
            output=result.stdout or "",
            max_chars=max_chars,
        )


class RunBash(Tool):
    name = "run_bash"
    description = """Run an arbitrary shell command and capture its combined stdout/stderr.

    When to use:
    - Compile code, start servers, install packages, or any CLI task
      not covered by other tools.

    Parameters:
    - command [string] REQUIRED - full shell command.

    Usage example:
    {
      "tool": "run_bash",
      "args": { "command": "ls -la | head" }
    }

    Special notes:
    - Executes with *shell=True* (/bin/bash -c on Unix).
    - Use absolute paths or prefix with `cd … &&` for other directories.
    - Output is captured and truncated to 20 kB in the return value.
    """
    inputs = {"command": {"type": "string", "description": "Command string."}}
    output_type = "string"

    def describe_call(self, *, command: str, **_: Any) -> str:
        return _clip(command, 80)

    def describe_result(self, result: Any, *, command: str, **_: Any) -> str:
        parsed = _parse_command_result(str(result))
        exit_code = parsed.get("exit_code")
        if exit_code is None:
            return _clip(str(result).splitlines()[0])
        cmd_label = _clip(command, 50)
        if exit_code == 0:
            return f"{cmd_label} succeeded"
        return f"{cmd_label} failed (exit {exit_code})"

    def build_result_details(self, result: Any, **_: Any) -> dict[str, Any] | None:
        parsed = _parse_command_result(str(result))
        if parsed.get("exit_code") in (None, 0):
            return None
        output = parsed.get("output") or ""
        return {
            "kind": "text",
            "title": "Shell output",
            "body": _tail(output) or "(no output)",
        }

    def forward(self, *, command: str) -> str:  # noqa: D401
        try:
            result = subprocess.run(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        except Exception as exc:
            return f"Error running command: {exc}"
        return _render_command_result(
            command_display=command,
            cwd=".",
            returncode=result.returncode,
            output=result.stdout or "",
            max_chars=20000,
        )


class Shell(Tool):
    name = "shell"
    description = """Run a shell command and return its combined stdout/stderr.

    When to use:
    - Execute system commands in a non-Windows environment.

    Parameters:
    - command             [array|string] REQUIRED - command argv list, or a command string.
    - workdir             [string]       OPTIONAL - working directory for the command.
    - timeout_ms          [number]       OPTIONAL - timeout in milliseconds.
    - stdin               [string]       OPTIONAL - text to send to stdin.
    - env                 [object]       OPTIONAL - environment vars to add/override.
    - shell_mode          [boolean]      OPTIONAL - if true, run string command via shell.
    - max_output_chars    [integer]      OPTIONAL - truncate return value to this size (default 20000).
    - sandbox_permissions [string]       OPTIONAL - accepts "require_escalated" or "use_default".
    - justification       [string]       OPTIONAL - human-readable reason for escalation.

    Usage example:
    {
      "tool": "shell",
      "args": {
        "command": ["bash", "-lc", "rg --files"],
        "workdir": ".",
        "timeout_ms": 5000
      }
    }

    Special notes:
    - This tool does not implement OS-level sandboxing.
    - Output is truncated to 20 kB in the return value.
    """
    inputs = {
        "command": {"type": "any", "description": "Command argv list or command string."},
        "workdir": {"type": "string", "description": "Working directory (optional).", "required": False},
        "timeout_ms": {"type": "number", "description": "Timeout in milliseconds (optional).", "required": False},
        "stdin": {"type": "string", "description": "Text to send to stdin (optional).", "required": False},
        "env": {"type": "object", "description": "Environment variables to add/override (optional).", "required": False},
        "shell_mode": {"type": "boolean", "description": "If true and command is a string, run via shell.", "required": False},
        "max_output_chars": {"type": "integer", "description": "Max characters to return (optional).", "required": False},
        "sandbox_permissions": {"type": "string", "description": "require_escalated or use_default.", "required": False},
        "justification": {"type": "string", "description": "Reason for escalation (optional).", "required": False},
    }
    output_type = "string"

    def describe_call(
        self,
        *,
        command: Union[str, list[str]],
        workdir: Optional[str] = None,
        **_: Any,
    ) -> str:
        cmd = _clip(_stringify_command(command), 90)
        if workdir:
            return f"{cmd} [cwd={workdir}]"
        return cmd

    def describe_result(self, result: Any, *, command: Union[str, list[str]], **_: Any) -> str:
        parsed = _parse_command_result(str(result))
        exit_code = parsed.get("exit_code")
        if exit_code is None:
            return _clip(str(result).splitlines()[0])
        cmd_label = _clip(_stringify_command(command), 50)
        if exit_code == 0:
            return f"{cmd_label} succeeded"
        return f"{cmd_label} failed (exit {exit_code})"

    def build_preview(
        self,
        *,
        command: Union[str, list[str]],
        workdir: Optional[str] = None,
        env: Optional[dict[str, Any]] = None,
        stdin: Optional[str] = None,
        **_: Any,
    ) -> dict[str, Any] | None:
        if not self.config.get("confirm_commands", False):
            return None
        lines = [f"Command: {_stringify_command(command)}", f"Working directory: {os.path.abspath(workdir or '.')}"]
        if env:
            lines.append("Environment overrides:")
            for key, value in env.items():
                lines.append(f"  {key}={value}")
        if stdin:
            lines.append("")
            lines.append("stdin:")
            lines.append(stdin)
        return {
            "kind": "text",
            "title": "Proposed shell command",
            "body": "\n".join(lines),
        }

    def build_result_details(self, result: Any, **_: Any) -> dict[str, Any] | None:
        parsed = _parse_command_result(str(result))
        exit_code = parsed.get("exit_code")
        output = parsed.get("output") or ""
        if exit_code == 0:
            return None
        return {
            "kind": "text",
            "title": "Shell output",
            "body": _tail(output) or "(no output)",
        }

    def needs_confirmation(self, **_: Any) -> bool:
        return bool(self.config.get("confirm_commands", False))

    def forward(
        self,
        *,
        command: Union[str, list[str]],
        workdir: Optional[str] = None,
        timeout_ms: Optional[float] = None,
        stdin: Optional[str] = None,
        env: Optional[dict[str, Any]] = None,
        shell_mode: bool = False,
        max_output_chars: Optional[int] = None,
        sandbox_permissions: str = "use_default",
        justification: Optional[str] = None,
    ) -> str:  # noqa: D401
        notes: list[str] = []
        if sandbox_permissions not in {"use_default", "require_escalated"}:
            return "Invalid sandbox_permissions: expected 'use_default' or 'require_escalated'."
        if sandbox_permissions == "require_escalated":
            notes.append("[note] sandbox_permissions=requested but this local Shell tool does not enforce sandbox escalation.")
        if justification:
            notes.append(f"[justification] {justification}")

        run_kwargs: dict[str, Any] = {"shell": False}
        command_display: str
        if isinstance(command, str):
            command_display = command
            if shell_mode:
                run_kwargs["shell"] = True
            else:
                import shlex
                command = shlex.split(command)
        elif isinstance(command, list) and all(isinstance(c, str) for c in command):
            if shell_mode:
                return "Invalid arguments: shell_mode=true requires command to be a string."
            command_display = " ".join(command)
        else:
            return "Invalid command: expected string or list[str]."

        if env is not None:
            if not isinstance(env, dict):
                return "Invalid env: expected object mapping env var names to values."
            bad_env = [k for k, v in env.items() if not isinstance(k, str) or not isinstance(v, (str, int, float, bool))]
            if bad_env:
                return "Invalid env: keys must be strings and values must be str/number/bool."
            env_map = os.environ.copy()
            env_map.update({k: str(v) for k, v in env.items()})
        else:
            env_map = None

        cwd = workdir or "."
        timeout_sec = None if timeout_ms is None else timeout_ms / 1000.0
        max_chars = 20000 if max_output_chars is None else max(1, int(max_output_chars))
        try:
            result = subprocess.run(
                command,
                cwd=cwd,
                env=env_map,
                input=stdin,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_sec,
                check=False,
                **run_kwargs,
            )
        except FileNotFoundError:
            if isinstance(command, list) and command:
                return f"Command not found: {command[0]}"
            return "Command not found."
        except subprocess.TimeoutExpired as exc:
            output = exc.output or ""
            output_str = output.decode("utf-8", errors="replace") if isinstance(output, bytes) else str(output)
            return _render_command_result(
                command_display=command_display,
                cwd=cwd,
                returncode=124,
                output=output_str,
                max_chars=max_chars,
                notes=notes,
                timed_out=True,
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            return f"Error running command: {exc}"

        return _render_command_result(
            command_display=command_display,
            cwd=cwd,
            returncode=result.returncode,
            output=result.stdout or "",
            max_chars=max_chars,
            notes=notes,
        )


class Delete(Tool):
    name = "delete"
    description = """Delete a file or directory (including all its contents).

    When to use:
    - Remove a file when it's no longer needed.
    - Clean up a directory and all of its contents.

    Parameters:
    - path [string] REQUIRED - target path (absolute or CWD‑relative) to delete.

    Usage example:
    {
      "tool": "delete",
      "args": { "path": "temp/old_data.txt" }
    }

    Special notes:
    - Use with caution, as deletion is permanent and irreversible.
    - Can delete both files and directories recursively.
    - Will report an error if the file or directory doesn't exist.
    - If confirm_edits is enabled, the agent will request confirmation before deletion.
    """
    inputs = {"path": {"type": "string", "description": "Path to file or directory to delete."}}
    output_type = "string"

    def describe_call(self, *, path: str, **_: Any) -> str:
        return path

    def build_preview(self, *, path: str, **_: Any) -> dict[str, Any] | None:
        if not os.path.exists(path):
            return None
        if os.path.isfile(path):
            item_type = "file"
        elif os.path.isdir(path):
            item_type = "directory"
        else:
            item_type = "path"
        details = [f"Path: {os.path.abspath(path)}", f"Type: {item_type}"]
        if os.path.isdir(path):
            try:
                details.append(f"Entries: {len(os.listdir(path))}")
            except Exception:
                pass
        return {
            "kind": "text",
            "title": f"Delete {path}",
            "body": "\n".join(details),
        }

    def needs_confirmation(self, **_: Any) -> bool:
        return bool(self.config.get("confirm_edits", False))

    def forward(self, *, path: str) -> str:  # noqa: D401
        if not os.path.exists(path):
            return f"Error: The path '{path}' does not exist."

        try:
            if os.path.isfile(path):
                os.remove(path)
                return f"Successfully deleted file: {path}"
            elif os.path.isdir(path):
                shutil.rmtree(path)
                return f"Successfully deleted directory and all its contents: {path}"
            else:
                return f"Error: '{path}' is neither a file nor a directory."
        except Exception as e:
            return f"Error deleting '{path}': {str(e)}"


class ListFiles(Tool):
    name = "list_files"
    description = """Return a directory listing with lightweight metadata.

    When to use:
    - Discover what files/folders exist before reading or editing.

    Parameters:
    - path           [string]  OPTIONAL - directory to list (defaults to current working directory).
    - include_hidden [boolean] OPTIONAL - include dotfiles (default false).
    - max_entries    [integer] OPTIONAL - cap the number of returned entries.

    Usage example:
    {
      "tool": "list_files",
      "args": { "path": "src" }
    }

    Output:
    - Success → array of strings with names plus metadata (type/size/line count).
    - Failure → string error message.

    Special notes:
    - Non‑recursive; call again on sub‑directories for deeper inspection.
    """
    inputs = {
        "path": {"type": "string", "description": "Directory path.", "required": False},
        "include_hidden": {"type": "boolean", "description": "Include dotfiles/directories (optional).", "required": False},
        "max_entries": {"type": "integer", "description": "Maximum entries to return (optional).", "required": False},
    }
    output_type = "array"

    def describe_call(self, *, path: str = ".", **_: Any) -> str:
        return path

    def describe_result(self, result: Any, *, path: str = ".", **_: Any) -> str:
        if isinstance(result, str):
            return result
        count = max(0, len(result) - 1)
        return f"Listed {count} entr{'y' if count == 1 else 'ies'} in {path}"

    def forward(
        self,
        *,
        path: str = ".",
        include_hidden: bool = False,
        max_entries: Optional[int] = None,
    ) -> list[str] | str:  # noqa: D401
        try:
            entries = os.listdir(path if path else ".")
            # Filter out hidden files (starting with '.') and agent_traces folders
            entries = [entry for entry in entries
                    if (include_hidden or not entry.startswith('.'))
                    and not entry.startswith('agent_traces')
                    and not entry.startswith('plan')
                    and not entry.startswith('reflection')]
            if max_entries is not None and max_entries < 1:
                return "Invalid max_entries: must be >= 1."

            # Directories first, then files
            dirs = sorted([entry for entry in entries if os.path.isdir(os.path.join(path, entry))])
            files = sorted([entry for entry in entries if os.path.isfile(os.path.join(path, entry))])
            ordered = dirs + files
            if max_entries is not None:
                ordered = ordered[:max_entries]

            def _fmt_size(num_bytes: int) -> str:
                units = ["B", "KB", "MB", "GB"]
                size = float(num_bytes)
                for unit in units:
                    if size < 1024 or unit == units[-1]:
                        return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
                    size /= 1024
                return f"{num_bytes} B"

            result: list[str] = []
            for entry in ordered:
                full = os.path.join(path, entry)
                if os.path.isdir(full):
                    try:
                        child_count = len(os.listdir(full))
                        result.append(f"{entry}/ [dir, {child_count} item{'s' if child_count != 1 else ''}]")
                    except Exception:
                        result.append(f"{entry}/ [dir]")
                    continue

                try:
                    size = os.path.getsize(full)
                except OSError:
                    size = -1

                line_info = "unknown lines"
                try:
                    with open(full, "r", encoding="utf-8") as fp:
                        line_count = sum(1 for _ in fp)
                    line_info = f"{line_count} line{'s' if line_count != 1 else ''}"
                except UnicodeDecodeError:
                    line_info = "binary"
                except Exception:
                    line_info = "unreadable"

                size_info = _fmt_size(size) if size >= 0 else "unknown size"
                result.append(f"{entry} [{line_info}, {size_info}]")

            hidden_note = "" if include_hidden else " (hidden excluded)"
            prefix = f". [{len(result)} entries shown]{hidden_note}"
            return [prefix] + result
        except FileNotFoundError:
            return f"Directory not found: {path}"
        except NotADirectoryError:
            return f"Not a directory: {path}"


class SearchFiles(Tool):
    name = "search_files"
    description = """Search file contents using ripgrep (`rg`).

    When to use:
    - Find where a symbol, string, or pattern appears before reading/editing files.
    - Quickly identify relevant files and line numbers.

    Parameters:
    - pattern        [string]  REQUIRED - regex pattern (or literal string if fixed_strings=true).
    - path           [string]  OPTIONAL - directory/file path to search (defaults to current directory).
    - glob           [string]  OPTIONAL - rg glob filter (e.g., "*.py").
    - max_results    [integer] OPTIONAL - maximum matching lines to return.
    - ignore_case    [boolean] OPTIONAL - case-insensitive search.
    - fixed_strings  [boolean] OPTIONAL - treat pattern as a literal string.
    - include_hidden [boolean] OPTIONAL - include hidden files/directories.

    Usage example:
    {
      "tool": "search_files",
      "args": { "pattern": "def main", "glob": "*.py", "max_results": 20 }
    }

    Special notes:
    - Requires `rg` (ripgrep) to be installed.
    - Returns line-numbered matches in `path:line:text` format.
    """
    inputs = {
        "pattern": {"type": "string", "description": "Regex pattern or literal string to search for."},
        "path": {"type": "string", "description": "Directory or file path to search (optional).", "required": False},
        "glob": {"type": "string", "description": "Ripgrep glob filter (optional), e.g. *.py", "required": False},
        "max_results": {"type": "integer", "description": "Maximum matching lines to return (optional).", "required": False},
        "ignore_case": {"type": "boolean", "description": "Case-insensitive search (optional).", "required": False},
        "fixed_strings": {"type": "boolean", "description": "Treat pattern literally (optional).", "required": False},
        "include_hidden": {"type": "boolean", "description": "Include hidden files and directories (optional).", "required": False},
    }
    output_type = "string"

    def describe_call(self, *, pattern: str, path: str = ".", **_: Any) -> str:
        return f"{pattern!r} in {path}"

    def describe_result(self, result: Any, **_: Any) -> str:
        text = str(result)
        first_line = text.splitlines()[0] if text else ""
        if first_line.startswith("[rg] no matches"):
            return first_line.replace("[rg] ", "").capitalize()
        if first_line.startswith("[rg] "):
            match_count = max(0, len(text.splitlines()) - 1)
            return f"Found {match_count} match(es)"
        return _clip(first_line or "Search completed")

    def forward(
        self,
        *,
        pattern: str,
        path: str = ".",
        glob: Optional[str] = None,
        max_results: Optional[int] = None,
        ignore_case: bool = False,
        fixed_strings: bool = False,
        include_hidden: bool = False,
    ) -> str:  # noqa: D401
        if not isinstance(pattern, str) or not pattern:
            return "Invalid pattern: expected non-empty string."
        if max_results is not None and max_results < 1:
            return "Invalid max_results: must be >= 1."

        cmd = ["rg", "--line-number", "--no-heading", "--color", "never"]
        if ignore_case:
            cmd.append("-i")
        if fixed_strings:
            cmd.append("-F")
        if include_hidden:
            cmd.append("--hidden")
        if glob:
            cmd.extend(["-g", glob])
        if max_results is not None:
            cmd.extend(["-m", str(max_results)])

        cmd.extend([pattern, path or "."])

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            return "Command not found: rg (install ripgrep to use search_files)."
        except Exception as exc:  # pragma: no cover - defensive guard
            return f"Error running rg: {exc}"

        output = result.stdout or ""
        if result.returncode == 0:
            header = f"[rg] {pattern!r} in {path or '.'}"
            if glob:
                header += f" (glob={glob})"
            return truncate(header + "\n" + output)
        if result.returncode == 1:
            header = f"[rg] no matches for {pattern!r} in {path or '.'}"
            if glob:
                header += f" (glob={glob})"
            return header
        return truncate(f"[rg exit code {result.returncode}]\n{output}")


class ReadPDF_txtimg(Tool):
    name = "read_pdf"
    description = """Extract text content and page images from PDF files.

    When to use:
    - Extract text from PDF documents
    - View full page images for PDFs with complex content like formulas, tables, or diagrams
    - Analyze scientific papers or technical documents with mixed content

    Parameters:
    - filename [string] REQUIRED - path to the PDF file
    - page     [integer] OPTIONAL - specific page number to extract (1-based indexing); omit for all pages
    - image_only [boolean] OPTIONAL - if true, only returns page images without text extraction

    Usage example:
    {
      "tool": "read_pdf",
      "args": { 
        "filename": "documents/paper.pdf",
        "page": 3,
        "image_only": true
      }
    }

    Special notes:
    - Returns both extracted text and base64-encoded page images
    - For complex content like math equations or diagrams, use image_only=true
    - Large PDFs may be limited to prevent excessive token usage
    """
    inputs = {
        "filename": {"type": "string", "description": "Path to the PDF file"},
        "page": {"type": "integer", "description": "Specific page to extract (1-based indexing, optional)", "required": False},
        "image_only": {"type": "boolean", "description": "If true, only returns page images without text", "required": False}
    }
    output_type = "object"

    def describe_call(self, *, filename: str, page: int = None, **_: Any) -> str:
        if page is None:
            return filename
        return f"{filename} page {page}"

    def describe_result(self, result: Any, *, filename: str, **_: Any) -> str:
        if isinstance(result, dict) and "error" in result:
            return str(result["error"])
        if isinstance(result, dict):
            text_count = len(result.get("text", []))
            img_count = len(result.get("images", []))
            return f"Read PDF {filename}: {text_count} text block(s), {img_count} image(s)"
        return super().describe_result(result, filename=filename)

    def forward(self, *, filename: str, page: int = None, image_only: bool = True) -> Dict[str, Any]:
        """
        Extract text and images from a PDF file
        
        Args:
            filename: Path to the PDF file
            page: Specific page to extract (1-based indexing); None for all pages
            image_only: If true, only returns page images without text extraction
            
        Returns:
            Dictionary with 'text' and 'images' keys
        """
        if not os.path.isfile(filename):
            return {"error": f"File not found: {filename}"}
            
        try:
            # Import the required libraries
            import pypdf
            from pdf2image import convert_from_path
            from PIL import Image
            import io
            
            result = {"text": [], "images": []}
            
            # Extract text using PyPDF if not image_only
            if not image_only:
                try:
                    pdf_reader = pypdf.PdfReader(filename)
                    
                    # Handle specific page or all pages
                    pages_to_process = [page-1] if page is not None else range(len(pdf_reader.pages))
                    
                    for page_num in pages_to_process:
                        if 0 <= page_num < len(pdf_reader.pages):
                            page_obj = pdf_reader.pages[page_num]
                            text = page_obj.extract_text()
                            result["text"].append({
                                "page": page_num + 1,
                                "content": text
                            })
                except Exception as e:
                    result["text_error"] = f"Error extracting text: {str(e)}"
            
            # Extract images using pdf2image
            try:
                # Convert specific page or all pages
                if page is not None:
                    images = convert_from_path(filename, first_page=page, last_page=page)
                else:
                    images = convert_from_path(filename)
                
                # Process each image
                for i, img in enumerate(images):
                    # Calculate the page number
                    page_number = page if page is not None else i + 1
                    
                    # # Resize image to a reasonable width while maintaining aspect ratio
                    # width, height = img.size
                    # new_width = min(1024, width)  # Limit width to 1024px max
                    # if width > new_width:
                    #     new_height = int(height * (new_width / width))
                    #     img = img.resize((new_width, new_height), Image.LANCZOS)
                    
                    # Convert to base64
                    buffer = io.BytesIO()
                    img.save(buffer, format="JPEG", quality=85)  # Use JPEG with good quality for smaller size
                    buffer.seek(0)
                    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
                    
                    result["images"].append({
                        "page": page_number,
                        "width": img.width,
                        "height": img.height,
                        "data": f"data:image/jpeg;base64,{img_base64}"
                    })
            except Exception as e:
                result["image_error"] = f"Error extracting images: {str(e)}"
            
            return result
            
        except ImportError as e:
            missing_lib = str(e).split("'")[1] if "'" in str(e) else str(e)
            return {
                "error": f"Missing required library: {missing_lib}. Please install with: pip install pypdf pdf2image pillow"
            }
        except Exception as e:
            return {"error": f"Error processing PDF: {str(e)}"}



class ReadPDF_notxt(Tool):
    name = "read_pdf"
    description = """Extract pages from PDF.

    When to use:
    - Extract text and images from PDF documents
    - View full page images for PDFs with complex content like formulas, tables, or diagrams
    - Analyze scientific papers or technical documents with mixed content

    Parameters:
    - filename [string] REQUIRED - path to the PDF file

    Usage example:
    {
      "tool": "read_pdf",
      "args": { 
        "filename": "documents/paper.pdf"
      }
    }

    Special notes:
    - Returns full-page base64-encoded page images
    """
    inputs = {
        "filename": {"type": "string", "description": "Path to the PDF file"},
        # "page": {"type": "integer", "description": "Specific page to extract (1-based indexing, optional)"},
        # "image_only": {"type": "boolean", "description": "If true, only returns page images without text"}
    }
    output_type = "object"

    def describe_call(self, *, filename: str, **_: Any) -> str:
        return filename

    def describe_result(self, result: Any, *, filename: str, **_: Any) -> str:
        if isinstance(result, dict) and "error" in result:
            return str(result["error"])
        if isinstance(result, dict):
            img_count = len(result.get("images", []))
            return f"Read PDF {filename}: {img_count} image(s)"
        return super().describe_result(result, filename=filename)

    def forward(self, *, filename: str) -> Dict[str, Any]:
        """
        Extract text and images from a PDF file
        
        Args:
            filename: Path to the PDF file
            
        Returns:
            Dictionary with 'images' key
        """
        page = None
        image_only = True
        if not os.path.isfile(filename):
            return {"error": f"File not found: {filename}"}
            
        try:
            # Import the required libraries
            import pypdf
            from pdf2image import convert_from_path
            from PIL import Image
            import io
            
            result = {"text": [], "images": []}
            
            # Extract text using PyPDF if not image_only
            if not image_only:
                try:
                    pdf_reader = pypdf.PdfReader(filename)
                    
                    # Handle specific page or all pages
                    pages_to_process = [page-1] if page is not None else range(len(pdf_reader.pages))
                    
                    for page_num in pages_to_process:
                        if 0 <= page_num < len(pdf_reader.pages):
                            page_obj = pdf_reader.pages[page_num]
                            text = page_obj.extract_text()
                            result["text"].append({
                                "page": page_num + 1,
                                "content": text
                            })
                except Exception as e:
                    result["text_error"] = f"Error extracting text: {str(e)}"
            
            # Extract images using pdf2image
            try:
                # Convert specific page or all pages
                if page is not None:
                    images = convert_from_path(filename, first_page=page, last_page=page)
                else:
                    images = convert_from_path(filename)
                
                # Process each image
                for i, img in enumerate(images):
                    # Calculate the page number
                    page_number = page if page is not None else i + 1
                    
                    # # Resize image to a reasonable width while maintaining aspect ratio
                    # width, height = img.size
                    # new_width = min(1024, width)  # Limit width to 1024px max
                    # if width > new_width:
                    #     new_height = int(height * (new_width / width))
                    #     img = img.resize((new_width, new_height), Image.LANCZOS)
                    
                    # Convert to base64
                    buffer = io.BytesIO()
                    img.save(buffer, format="JPEG", quality=85)  # Use JPEG with good quality for smaller size
                    buffer.seek(0)
                    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
                    
                    result["images"].append({
                        "page": page_number,
                        "width": img.width,
                        "height": img.height,
                        "data": f"data:image/jpeg;base64,{img_base64}"
                    })
            except Exception as e:
                result["image_error"] = f"Error extracting images: {str(e)}"
            
            return result
            
        except ImportError as e:
            missing_lib = str(e).split("'")[1] if "'" in str(e) else str(e)
            return {
                "error": f"Missing required library: {missing_lib}. Please install with: pip install pypdf pdf2image pillow"
            }
        except Exception as e:
            return {"error": f"Error processing PDF: {str(e)}"}

ReadPDF = ReadPDF_notxt

class ViewImage(Tool):
    """
    Encode a local image as a data-URI so the model can inspect it.
    Automatically resizes the image to 512px width while maintaining aspect ratio.
    """
    name = "view_image"
    description = """Load a local image, resize it to 512px width, and return a base64 data‑URI so the model can inspect it.

    When to use:
    - Show the model screenshots, plots, or photos stored on disk.

    Parameters:
    - filename [string] REQUIRED - path to PNG, JPG, or similar.

    Usage example:
    {
      "tool": "view_image",
      "args": { "filename": "output/plot.png" }
    }

    Special notes:
    - Images are automatically resized to 512px width while maintaining aspect ratio.
    - Large images may be truncated by upstream token limits.
    """
    inputs = {
        "filename": {
            "type": "string",
            "description": "Path to the image file (PNG/JPG/…)."
        }
    }
    output_type = "image"

    def describe_call(self, *, filename: str, **_: Any) -> str:
        return filename

    def describe_result(self, result: Any, *, filename: str, **_: Any) -> str:
        if isinstance(result, str) and result.startswith("data:image"):
            return f"Loaded image {filename}"
        return str(result)

    def forward(self, *, filename: str) -> str:  # noqa: D401
        if not os.path.isfile(filename):
            return f"File not found: {filename}"
        
        # try:
        #     # Try to import PIL for image resizing
        #     from PIL import Image
        #     import io
            
        #     # Open the image and resize to 512px width, maintaining aspect ratio
        #     img = Image.open(filename)
        #     width, height = img.size
        #     new_width = 512
        #     new_height = int(height * (new_width / width))
        #     img = img.resize((new_width, new_height), Image.LANCZOS)
            
        #     # Save to a BytesIO buffer
        #     buffer = io.BytesIO()
        #     img.save(buffer, format=img.format or "PNG")
        #     buffer.seek(0)
        #     image_data = buffer.read()
            
        #     # Get mime type
        #     mime, _ = mimetypes.guess_type(filename)
        #     if mime is None:
        #         mime = f"image/{img.format.lower()}" if img.format else "image/png"
            
        #     # Encode to base64
        #     data64 = base64.b64encode(image_data).decode()
            
        # except ImportError:
        #     # Fallback if PIL is not available
        #     LOGGER.warning("PIL not available, serving original image without resizing")
        mime, _ = mimetypes.guess_type(filename)
        if mime is None:
            mime = "image/png"
        with open(filename, "rb") as fp:
            data64 = base64.b64encode(fp.read()).decode()
        
        return f"data:{mime};base64,{data64}"


class UpdatePlan(Tool):
    name = "update_plan"
    description = """Update the task plan.

    When to use:
    - Create or update a short step-by-step plan.
    - Mark steps as pending, in_progress, or completed.

    Parameters:
    - plan [array] REQUIRED - list of plan items with "step" and "status".
    - explanation [string] OPTIONAL - short reason or context for the update.

    Usage example:
    {
      "tool": "update_plan",
      "args": {
        "explanation": "Refined steps after reviewing files",
        "plan": [
          {"step": "Inspect config", "status": "completed"},
          {"step": "Adjust parser", "status": "in_progress"},
          {"step": "Run tests", "status": "pending"}
        ]
      }
    }

    Special notes:
    - At most one step should be in_progress.
    """
    inputs = {
        "plan": {"type": "array", "description": "List of {step, status} items."},
        "explanation": {"type": "string", "description": "Optional explanation for the update.", "required": False},
    }
    output_type = "string"

    def describe_call(self, **_: Any) -> str:
        return "plan"

    def build_result_details(self, result: Any, **_: Any) -> dict[str, Any] | None:
        last_plan = getattr(self, "_last_plan", None)
        if not last_plan:
            return None
        return {
            "kind": "plan",
            "title": "Plan",
            "plan": last_plan.get("plan", []),
            "explanation": last_plan.get("explanation"),
        }

    def forward(self, *, plan: Any, explanation: Optional[str] = None) -> str:  # noqa: D401
        if isinstance(plan, str):
            try:
                plan = json.loads(plan)
            except json.JSONDecodeError:
                return "Invalid plan: expected JSON array or list."

        if not isinstance(plan, list):
            return "Invalid plan: expected list."

        allowed = {"pending", "in_progress", "completed"}
        in_progress = 0
        normalized: list[dict[str, str]] = []

        for item in plan:
            if isinstance(item, str):
                step = item
                status = "pending"
            elif isinstance(item, dict):
                step = item.get("step")
                status = item.get("status", "pending")
            else:
                return "Invalid plan item: expected string or object."

            if not isinstance(step, str) or not step.strip():
                return "Invalid plan item: missing step."
            if not isinstance(status, str) or status not in allowed:
                return f"Invalid plan item status: {status!r}."

            if status == "in_progress":
                in_progress += 1

            normalized.append({"step": step, "status": status})

        if in_progress > 1:
            return "Invalid plan: at most one step can be in_progress."

        self._last_plan = {"explanation": explanation, "plan": normalized}
        return "Plan updated"


class MakePlan(Tool):
    name = "make_plan"
    description = """Write a numbered plain‑text multi‑step plan to disk.

    When to use:
    - At the start of a complex task to outline the intended steps.

    Parameters:
    - content [string] REQUIRED - plan body; numbering auto‑increments.

    Usage example:
    {
      "tool": "make_plan",
      "args": { "content": "1. Fetch data\\n2. Clean data\\n3. Train model" }
    }

    Special notes:
    - Files are named *plan_XX.txt* where *XX* increments safely.
    """
    inputs = {"content": {"type": "string", "description": "Plan body."}}
    output_type = "string"

    def describe_call(self, **_: Any) -> str:
        return "plan"

    def forward(self, *, content: str) -> str:  # noqa: D401
        idx = 0
        while True:
            fname = f"plan_{idx:02d}.txt"
            if not os.path.exists(fname):
                break
            idx += 1
        with open(fname, "w", encoding="utf-8") as fp:
            fp.write(content)
        return f"Plan saved → {fname}"


class Reflect(Tool):
    name = "reflect"
    description = """Document the agent's reflections on observations before proceeding to the next step.

    When to use:
    - Between tool steps to analyze observations and determine the next course of action
    - When encountering unexpected results that require deeper analysis
    - To document reasoning about complex decisions before proceeding

    Parameters:
    - content [string] REQUIRED - reflection text containing analysis of observations and reasoning.

    Usage example:
    {
      "tool": "reflect",
      "args": { "content": "After examining the output, I notice that:\\n1. The error occurs in the data loading step\\n2. The file format appears to be incompatible\\n\\nThis suggests we need to implement a different parsing approach." }
    }

    Special notes:
    - Files are named *reflection_XX.txt* where *XX* increments safely.
    - Helps maintain a record of reasoning and decision-making process.
    """
    inputs = {"content": {"type": "string", "description": "Reflection content."}}
    output_type = "string"

    def describe_call(self, **_: Any) -> str:
        return "reflection"

    def forward(self, *, content: str) -> str:  # noqa: D401
        idx = 0
        while True:
            fname = f"reflection_{idx:02d}.txt"
            if not os.path.exists(fname):
                break
            idx += 1
        with open(fname, "w", encoding="utf-8") as fp:
            fp.write(content)
        return f"Reflection saved → {fname}"


class FinalAnswer(Tool):
    name = "final_answer"
    description = """Return the final answer to the user - **must** be the last tool call.

    When to use:
    - All tasks are complete and you need to deliver the end result.

    Parameters:
    - answer [string] REQUIRED - text to present to the user.

    Usage example:
    {
      "tool": "final_answer",
      "args": { "answer": "All tasks completed successfully." }
    }

    Special notes:
    - No further tool calls are allowed after this.
    """
    inputs = {"answer": {"type": "string", "description": "Answer text."}}
    output_type = "string"

    def describe_call(self, **_: Any) -> str:
        return "answer"

    def describe_result(self, result: Any, **_: Any) -> str:
        return "Final answer ready"

    def forward(self, *, answer: str) -> str:  # noqa: D401
        return answer


class GetUserInput(Tool):
    name = "get_user_input"
    description = """Prompt the human user and wait for text input.

    When to use:
    - A critical parameter is unknown and cannot be inferred.

    Parameters:
    - message [string] REQUIRED - prompt shown to the user.

    Usage example:
    {
      "tool": "get_user_input",
      "args": { "message": "Enter your API key:" }
    }

    Special notes:
    - Execution blocks until the user responds in the console.
    """
    inputs = {"message": {"type": "string", "description": "Prompt shown to the user."}}
    output_type = "string"

    def describe_call(self, **_: Any) -> str:
        return "prompt"

    def describe_result(self, result: Any, **_: Any) -> str:
        return f"User input received: {_clip(str(result), 60)}"

    def forward(self, *, message: str) -> str:  # noqa: D401
        return input(message + " ")
