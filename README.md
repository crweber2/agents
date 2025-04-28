# Autonomous Scientific Computing Agent Framework

A specialized framework for creating autonomous agents that can solve complex scientific computing problems through interaction with the OpenAI Chat API and tool-calling capabilities.

## Overview

This framework provides a powerful Agent class designed to tackle scientific computing challenges in areas such as physics, mathematics, orbital mechanics, data analysis, and numerical simulations. It uses a variety of tools to read data, write and execute Python code, analyze results, and generate visualizations. The agent can work independently or with managed sub-agents to solve mathematically complex problems.

## Installation

### Prerequisites

- Python 3.8+ 
- OpenAI API key (set as environment variable `OPENAI_API_KEY`)

### Setup

```bash
# Clone the repository
git clone https://github.com/crweber2/agents.git
cd agents

# Install dependencies
pip install openai rich
```

## Project Structure

- **agents.py**: Core Agent class implementation with memory management and LLM interaction
- **agent_tools.py**: Collection of tools used by the Agent (file operations, execution, etc.)
- **__init__.py**: Package imports and public API surface
- **run_agents.py**: CLI entrypoint to execute agents on specific tasks
- **run_validation.py**: Test runner for validation problems
- **test_problems/**: Directory containing various test problems
  - **easy/**: Simple problems for basic testing
  - **medium/**: Moderately complex problems
  - **hard/**: Challenging problems requiring advanced reasoning
  - **pytorch/**: Problems involving PyTorch usage
  - **validation/**: Problems with known solutions for automated testing

## Usage

### Basic Usage

```python
from agents import Agent, LLMClient
from agent_tools import WriteFile, ReadFile, RunPython, RunBash

# Create an LLM client
model = LLMClient(model_id="gpt-4.1")

# Create an agent with tools
agent = Agent(
    tools=[WriteFile(), ReadFile(), RunPython(), RunBash()],
    model=model,
    name="physics_agent",
    max_steps=20
)

# Run the agent on a scientific computing task
result = agent.run("Simulate a damped harmonic oscillator and plot the position vs time")
print(result)
```

### Command Line Interface

```bash
# Run an agent on a scientific computing task
python run_agents.py "Solve the heat equation in 1D with Dirichlet boundary conditions"

# Run with debug logging
python run_agents.py -d "Analyze earthquake data in earthquake.csv using FFT"

# Use a local LLM
python run_agents.py -l "Calculate orbital parameters for a satellite with J2 perturbation"

# Require confirmation before editing or deleting files
python run_agents.py -c "Implement a Runge-Kutta method for solving ODEs"
```

## Testing and Validation

The framework includes a validation system to test the agent against problems with known solutions.

### Test Problems

- The `test_problems` directory contains various scientific computing problems organized by difficulty
- Problems include physics simulations, differential equations, data analysis, and more:
  - Orbital mechanics (solar system simulations, satellite tracking)
  - Fluid dynamics (shock tubes, advection problems)
  - Signal processing (FFT analysis of earthquake data)
  - Energy systems (PV battery optimization, wind turbine modeling)
  - Machine learning (Gaussian clustering, LSTM forecasting)
- Each problem is presented as a text file with detailed scientific specifications
- Validation problems have corresponding solution files with expected numerical outputs

### Running Validation

```bash
# Run validation tests on all validation problems
python run_validation.py

# This will:
# - Run each validation problem multiple times
# - Compare agent outputs with expected solutions
# - Report success rates for each problem
```

The validation system is useful for:
- Testing agent capabilities
- Ensuring reliability across different types of problems
- Benchmarking improvements to the agent framework

## Author

Chris Weber, crweber@gmail.com
