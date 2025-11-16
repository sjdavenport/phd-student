# Simulation Agent

An LLM-powered agent that automatically creates and runs statistical simulations for research papers.

## Features

- **Autonomous Simulation Creation**: Describe what you want to simulate in natural language
- **Automatic Code Generation**: The agent writes Python simulation code
- **Self-Debugging**: Automatically fixes bugs and retries
- **Output Validation**: Checks results for statistical sanity
- **Result Synthesis**: Generates human-readable reports

## Installation

```bash
pip install -r requirements.txt
```

Set your Anthropic API key:
```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

## Quick Start

```python
from simulation_agent import SimulationAgent, AnthropicBackend

# Initialize
llm = AnthropicBackend()
agent = SimulationAgent(llm=llm)

# Define a task
task = """
Simulate the Type I error rate of a two-sample t-test when data
comes from a log-normal distribution. Test with sample sizes 10, 30, 100.
"""

# Run the simulation
result = agent.run(task)

if result["success"]:
    print(result["report"])
    print(result["results"])
```

## Architecture

The agent follows this workflow:

1. **Task Parser** - Extracts structured requirements from natural language
2. **Simulation Designer** - Creates a simulation strategy
3. **Code Generator** - Writes Python code
4. **Execution Engine** - Runs code in sandboxed environment
5. **Output Validator** - Checks results for correctness
6. **Debug Loop** - Fixes errors and retries (up to max_retries)
7. **Result Synthesizer** - Generates final report

## Configuration

```python
agent = SimulationAgent(
    llm=llm,
    max_retries=5,          # Max debug attempts
    timeout_seconds=300,    # Execution timeout
    verbose=True            # Print progress
)
```

## Supported LLM Backends

- **AnthropicBackend**: Claude models (default: claude-sonnet-4-20250514)
- **OpenAIBackend**: GPT models (default: gpt-4)

## Example Tasks

```python
# Type I error under non-normality
"Simulate Type I error rate for t-test with skewed data"

# Power analysis
"Simulate the power of a paired t-test for effect sizes 0.2, 0.5, 0.8"

# Bootstrap confidence intervals
"Compare coverage of bootstrap vs normal CI for the median"

# Regression assumptions
"Simulate bias in OLS when errors are heteroscedastic"
```

## Safety

- Code runs in subprocess (isolated from main process)
- Configurable timeout prevents infinite loops
- No file I/O in generated code
- Resource limits enforced
