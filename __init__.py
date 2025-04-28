"""
Autonomous Coding Agent Package.

This package provides an Agent class and tools for creating autonomous coding agents.

Author: Chris Weber, crweber@gmail.com
"""

# Import Agent from agents.py
from agents import Agent, LLMClient

# Import tools from agent_tools.py
from agent_tools import (
    Tool, WriteFile, ReadFile, EditFile, Delete, RunPython,
    RunBash, ViewImage, ListFiles, MakePlan, FinalAnswer, GetUserInput
)
