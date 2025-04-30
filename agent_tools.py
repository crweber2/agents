"""
Agent Tools Module.

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
- MakePlan: Create step-by-step plans
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
    - filename [string] REQUIRED - path to the file.

    Usage example:
    {
      "tool": "read_file",
      "args": { "filename": "src/utils.py" }
    }

    Special notes:
    - Binary or very large files (>20 kB) are truncated with a notice.
    - If the user already pasted the file contents, reuse that copy
      instead of reading the file again.
    """
    inputs = {"filename": {"type": "string", "description": "Path."}}
    output_type = "string"

    def forward(self, *, filename: str) -> str:  # noqa: D401
        if not os.path.isfile(filename):
            return f"File not found: {filename}"
        try:
            with open(filename, "r", encoding="utf-8") as fp:
                content = fp.read()
                return truncate(content)
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

    def forward(self, *, filename: str, search_string: str, replace_string: str) -> str:  # noqa: D401
        """
        Show a colored unified diff of the proposed changes, optionally ask for approval,
        and apply the changes if approved or if confirmation is disabled.
        """
        if not os.path.isfile(filename):
            return f"File not found: {filename}"
        with open(filename, "r", encoding="utf-8") as fp:
            original = fp.read()
        patched = original.replace(search_string, replace_string)
        if original == patched:
            return "No matches - nothing changed."
            
        # Show diff when verbose
        if LOGGER.level <= logging.INFO:
            import difflib
            diff = difflib.unified_diff(
                original.splitlines(), patched.splitlines(),
                fromfile=filename, tofile=f"{filename} (proposed)",
                lineterm=""
            )
            diff_text = "\n".join(diff)
            try:
                # Try to use rich for colored output if available
                try:
                    from rich.console import Console
                    from rich.syntax import Syntax
                    console = Console()
                    console.print(Syntax(diff_text, "diff", theme="ansi_dark"))
                except ImportError:
                    print(diff_text)
                    
                # Interactive confirmation (if enabled)
                confirm_edits = self.config.get("confirm_edits", False)
                if confirm_edits:
                    ans = input("Apply these changes? [y/N] ").strip().lower()
                    if ans != "y":
                        feedback = input("(Optional) feedback for the agent, or just press ↵ to skip: ")
                        return f"Edit rejected by user. Feedback: {feedback or '<none>'}"
            except Exception as e:
                LOGGER.error(f"Error displaying diff: {e}")
        
        # Apply changes
        with open(filename, "w", encoding="utf-8") as fp:
            fp.write(patched)
        return f"Replaced {original.count(search_string)} occurrence(s) in {filename}"


class RunPython(Tool):
    name = "run_python"
    description = """Execute a Python script in a subprocess and stream stdout/stderr live.

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
    - Output is streamed in real‑time; the return value is truncated to the first 20 kB.
    - If execution exceeds max_time, the process is terminated with a timeout message.
    - Command-line arguments are passed as-is to the script.
    """
    inputs = {
        "filename": {"type": "string", "description": "Script path."},
        "args": {"type": "string", "description": "Command-line arguments to pass to the script (optional)."},
        "max_time": {"type": "number", "description": "Maximum execution time in seconds (optional)."}
    }
    output_type = "string"

    def forward(self, *, filename: str, args: str = "", max_time: float = 300) -> str:  # noqa: D401
        import time, signal, threading
        
        if not os.path.isfile(filename):
            return f"File not found: {filename}"
        
        # Build command with optional arguments
        cmd = [sys.executable, filename]
        if args:
            # Split the args string properly to handle quoted arguments
            import shlex
            cmd.extend(shlex.split(args))
            
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        output: list[str] = []
        terminated = False
        
        # Set up timer to kill process if it exceeds max_time
        if max_time is not None:
            def kill_process():
                nonlocal terminated
                terminated = True
                output.append(f"\n[TIMEOUT] Process terminated after {max_time} seconds\n")
                # Try SIGTERM first for graceful shutdown
                try:
                    proc.terminate()
                    # Give it a moment to terminate gracefully
                    time.sleep(0.5)
                    if proc.poll() is None:  # If still running
                        proc.kill()  # Force kill
                except Exception:
                    # If terminate fails, try to force kill
                    try:
                        proc.kill()
                    except Exception:
                        pass
                        
            # Set up timer
            timer = threading.Timer(max_time, kill_process)
            timer.daemon = True  # Don't let timer block program exit
            timer.start()
        
        try:
            assert proc.stdout is not None
            for line in proc.stdout:  # pragma: no cover - interactive run
                print(line, end="")
                output.append(line)
                # Check if process was terminated by timer
                if terminated:
                    break
            proc.wait(timeout=0.5)  # Small timeout for final wait
        except subprocess.TimeoutExpired:
            # This happens if proc.wait times out
            pass
        finally:
            # Clean up timer if it's still active
            if max_time is not None:
                timer.cancel()
            # Ensure process is terminated
            if proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=0.5)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
        
        return truncate("".join(output))


class RunBash(Tool):
    name = "run_bash"
    description = """Run an arbitrary shell command and stream its combined stdout/stderr.

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
    - Output is truncated to 20 kB in the return value.
    """
    inputs = {"command": {"type": "string", "description": "Command string."}}
    output_type = "string"

    def forward(self, *, command: str) -> str:  # noqa: D401
        proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        output: list[str] = []
        assert proc.stdout is not None
        for line in proc.stdout:  # pragma: no cover - interactive run
            print(line, end="")
            output.append(line)
        proc.wait()
        return truncate("".join(output))


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
    - If CONFIRM_EDITS is enabled, will prompt for confirmation before deletion.
    """
    inputs = {"path": {"type": "string", "description": "Path to file or directory to delete."}}
    output_type = "string"

    def forward(self, *, path: str) -> str:  # noqa: D401
        if not os.path.exists(path):
            return f"Error: The path '{path}' does not exist."
        
        item_type = "file" if os.path.isfile(path) else "directory with all contents" if os.path.isdir(path) else "unknown item"
        
        # Interactive confirmation (if enabled)
        confirm_edits = self.config.get("confirm_edits", False)
        if confirm_edits:
            ans = input(f"Delete {item_type} at '{path}'? [y/N] ").strip().lower()
            if ans != "y":
                feedback = input("(Optional) feedback for the agent, or just press ↵ to skip: ")
                return f"Deletion rejected by user. Feedback: {feedback or '<none>'}"
            
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
    description = """Return a directory listing.

    When to use:
    - Discover what files/folders exist before reading or editing.

    Parameters:
    - path [string] OPTIONAL - directory to list (defaults to current working directory).

    Usage example:
    {
      "tool": "list_files",
      "args": { "path": "src" }
    }

    Output:
    - Success → array of file/dir names.
    - Failure → string error message.

    Special notes:
    - Non‑recursive; call again on sub‑directories for deeper inspection.
    """
    inputs = {"path": {"type": "string", "description": "Directory path."}}
    output_type = "array"

    def forward(self, *, path: str = ".") -> list[str] | str:  # noqa: D401
        try:
            entries = os.listdir(path if path else ".")
            # Filter out hidden files (starting with '.') and agent_traces folders
            entries = [entry for entry in entries 
                      if not entry.startswith('.') and not entry.startswith('agent_traces')]
            
            result = []
            
            # First add directories with trailing slash
            dirs = sorted([f"{entry}/" for entry in entries 
                          if os.path.isdir(os.path.join(path, entry))])
            
            # Then add files (without slash)
            files = sorted([entry for entry in entries 
                           if os.path.isfile(os.path.join(path, entry))])
            
            # Combine, with directories first
            result = dirs + files
            return result
        except FileNotFoundError:
            return f"Directory not found: {path}"


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

    def forward(self, *, filename: str) -> str:  # noqa: D401
        if not os.path.isfile(filename):
            return f"File not found: {filename}"
        
        try:
            # Try to import PIL for image resizing
            from PIL import Image
            import io
            
            # Open the image and resize to 512px width, maintaining aspect ratio
            img = Image.open(filename)
            width, height = img.size
            new_width = 512
            new_height = int(height * (new_width / width))
            img = img.resize((new_width, new_height), Image.LANCZOS)
            
            # Save to a BytesIO buffer
            buffer = io.BytesIO()
            img.save(buffer, format=img.format or "PNG")
            buffer.seek(0)
            image_data = buffer.read()
            
            # Get mime type
            mime, _ = mimetypes.guess_type(filename)
            if mime is None:
                mime = f"image/{img.format.lower()}" if img.format else "image/png"
            
            # Encode to base64
            data64 = base64.b64encode(image_data).decode()
            
        except ImportError:
            # Fallback if PIL is not available
            LOGGER.warning("PIL not available, serving original image without resizing")
            mime, _ = mimetypes.guess_type(filename)
            if mime is None:
                mime = "image/png"
            with open(filename, "rb") as fp:
                data64 = base64.b64encode(fp.read()).decode()
        
        return f"data:{mime};base64,{data64}"

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

    def forward(self, *, message: str) -> str:  # noqa: D401
        return input(message + " ")
