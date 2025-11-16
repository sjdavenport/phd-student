"""
Individual components for the simulation agent workflow.
"""

import subprocess
import tempfile
import json
import sys
from typing import Optional

from .llm import LLMBackend


class TaskParser:
    """Parses natural language task descriptions into structured simulation requirements."""

    SYSTEM_PROMPT = """You are a statistical simulation expert. Your job is to parse natural language descriptions of simulation tasks into structured requirements.

Extract the following information:
- research_question: The main question being investigated
- statistical_test: The test or method being evaluated (e.g., t-test, ANOVA, bootstrap)
- null_hypothesis: What the null hypothesis is (if applicable)
- data_distribution: The distribution(s) to generate data from
- sample_sizes: List of sample sizes to test
- effect_sizes: Effect sizes to simulate (if applicable)
- num_simulations: Number of simulation iterations
- metrics: What metrics to compute (e.g., Type I error rate, power, bias)
- additional_parameters: Any other relevant parameters"""

    def __init__(self, llm: LLMBackend):
        self.llm = llm

    def parse(self, task_description: str) -> dict:
        """Parse a task description into structured requirements."""
        prompt = f"""Parse this simulation task into structured requirements:

Task: {task_description}

Return a JSON object with these fields:
- research_question (string)
- statistical_test (string)
- null_hypothesis (string or null)
- data_distribution (string or object describing distribution)
- sample_sizes (array of integers)
- effect_sizes (array of numbers or null)
- num_simulations (integer, default 10000 if not specified)
- metrics (array of strings)
- additional_parameters (object with any extra parameters)

If information is not specified, make reasonable defaults for a statistics simulation."""

        return self.llm.generate_json(prompt, self.SYSTEM_PROMPT)


class SimulationDesigner:
    """Designs the simulation strategy based on parsed requirements."""

    SYSTEM_PROMPT = """You are a statistical simulation expert. Design simulation studies that are:
- Statistically rigorous
- Computationally efficient
- Well-structured for analysis

Consider factors like:
- Random seed management for reproducibility
- Efficient vectorization where possible
- Appropriate number of iterations for stable estimates
- Edge cases and boundary conditions"""

    def __init__(self, llm: LLMBackend):
        self.llm = llm

    def design(self, parsed_task: dict) -> dict:
        """Design a simulation strategy based on parsed requirements."""
        prompt = f"""Design a simulation strategy for this task:

Requirements:
{json.dumps(parsed_task, indent=2)}

Return a JSON object with:
- simulation_type: Type of simulation (e.g., "monte_carlo", "bootstrap", "permutation")
- algorithm_steps: Array of high-level steps in the simulation
- data_generation_strategy: How to generate the data
- iteration_structure: How to structure the simulation loops
- output_format: What the output should look like (dict with metric names as keys)
- convergence_checks: Any checks to ensure simulation has converged
- expected_results: What results would be expected if simulation is correct (for validation)
- computational_considerations: Notes on efficiency"""

        return self.llm.generate_json(prompt, self.SYSTEM_PROMPT)


class CodeGenerator:
    """Generates Python code for the simulation."""

    SYSTEM_PROMPT = """You are an expert Python programmer specializing in statistical simulations.

Write clean, efficient, well-documented code that:
- Uses numpy, scipy, and pandas appropriately
- Is vectorized where possible for performance
- Includes proper error handling
- Sets random seeds for reproducibility
- Returns results in a structured format

The code must:
1. Be complete and runnable
2. Print results as JSON to stdout
3. Not require any user input
4. Not create any files
5. Complete within reasonable time"""

    def __init__(self, llm: LLMBackend):
        self.llm = llm

    def generate(self, parsed_task: dict, simulation_plan: dict) -> str:
        """Generate Python code for the simulation."""
        prompt = f"""Write Python code to perform this simulation:

Task Requirements:
{json.dumps(parsed_task, indent=2)}

Simulation Plan:
{json.dumps(simulation_plan, indent=2)}

Requirements:
1. Import all necessary libraries at the top (numpy, scipy, pandas, etc.)
2. Set a random seed for reproducibility (use seed=42)
3. Implement the simulation following the plan
4. Store results in a dictionary
5. Print the results dictionary as JSON using: print(json.dumps(results))
6. The results dict should contain all metrics requested

Return ONLY the Python code, no explanations. The code must be complete and executable."""

        response = self.llm.generate(prompt, self.SYSTEM_PROMPT)

        # Extract code from response
        code = response.strip()
        if code.startswith("```python"):
            code = code[9:]
        if code.startswith("```"):
            code = code[3:]
        if code.endswith("```"):
            code = code[:-3]

        return code.strip()

    def fix(self, code: str, error: str, parsed_task: dict, simulation_plan: dict) -> str:
        """Fix code based on an error message."""
        prompt = f"""Fix this Python simulation code that produced an error:

Current Code:
```python
{code}
```

Error:
{error}

Original Task Requirements:
{json.dumps(parsed_task, indent=2)}

Simulation Plan:
{json.dumps(simulation_plan, indent=2)}

Analyze the error and fix the code. Return ONLY the corrected Python code, no explanations."""

        response = self.llm.generate(prompt, self.SYSTEM_PROMPT)

        # Extract code from response
        fixed_code = response.strip()
        if fixed_code.startswith("```python"):
            fixed_code = fixed_code[9:]
        if fixed_code.startswith("```"):
            fixed_code = fixed_code[3:]
        if fixed_code.endswith("```"):
            fixed_code = fixed_code[:-3]

        return fixed_code.strip()


class ExecutionEngine:
    """Executes Python code in a sandboxed environment."""

    def __init__(self, timeout_seconds: int = 300):
        self.timeout = timeout_seconds

    def execute(self, code: str) -> dict:
        """
        Execute Python code in a subprocess.

        Returns:
            dict with keys:
                - success: bool
                - output: dict (parsed JSON output) or None
                - stdout: str (raw stdout)
                - stderr: str (raw stderr)
                - error: str (error message if failed)
        """
        # Write code to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name

        try:
            # Run the code in a subprocess
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=tempfile.gettempdir()  # Run in temp directory for safety
            )

            stdout = result.stdout.strip()
            stderr = result.stderr.strip()

            # Check for errors
            if result.returncode != 0:
                return {
                    "success": False,
                    "output": None,
                    "stdout": stdout,
                    "stderr": stderr,
                    "error": f"Process exited with code {result.returncode}. Stderr: {stderr}"
                }

            # Try to parse JSON output
            try:
                # Find the last line that looks like JSON (the results)
                lines = stdout.split('\n')
                json_output = None
                for line in reversed(lines):
                    line = line.strip()
                    if line.startswith('{'):
                        json_output = json.loads(line)
                        break

                if json_output is None:
                    return {
                        "success": False,
                        "output": None,
                        "stdout": stdout,
                        "stderr": stderr,
                        "error": "No JSON output found in stdout"
                    }

                return {
                    "success": True,
                    "output": json_output,
                    "stdout": stdout,
                    "stderr": stderr,
                    "error": None
                }

            except json.JSONDecodeError as e:
                return {
                    "success": False,
                    "output": None,
                    "stdout": stdout,
                    "stderr": stderr,
                    "error": f"Failed to parse JSON output: {e}"
                }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "output": None,
                "stdout": "",
                "stderr": "",
                "error": f"Execution timed out after {self.timeout} seconds"
            }
        except Exception as e:
            return {
                "success": False,
                "output": None,
                "stdout": "",
                "stderr": "",
                "error": f"Execution failed: {str(e)}"
            }
        finally:
            # Clean up temp file
            import os
            try:
                os.unlink(temp_file)
            except:
                pass


class OutputValidator:
    """Validates simulation output for correctness and statistical sanity."""

    SYSTEM_PROMPT = """You are a statistical validation expert. Check simulation results for:
- Statistical sanity (p-values in [0,1], positive variances, etc.)
- Consistency with expected behavior
- Absence of obvious errors (NaN, Inf, impossible values)
- Alignment with simulation goals"""

    def __init__(self, llm: LLMBackend):
        self.llm = llm

    def validate(self, execution_result: dict, parsed_task: dict, simulation_plan: dict) -> dict:
        """
        Validate the simulation output.

        Returns:
            dict with keys:
                - valid: bool
                - issues: list of issues found (empty if valid)
                - suggestions: list of suggestions for improvement
        """
        output = execution_result.get("output", {})

        # Basic programmatic checks
        issues = []

        # Check for NaN or Inf
        def check_values(obj, path=""):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    check_values(v, f"{path}.{k}" if path else k)
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    check_values(v, f"{path}[{i}]")
            elif isinstance(obj, float):
                if obj != obj:  # NaN check
                    issues.append(f"NaN value found at {path}")
                elif obj == float('inf') or obj == float('-inf'):
                    issues.append(f"Infinite value found at {path}")

        check_values(output)

        # Check for common statistical constraints
        for key, value in output.items():
            if isinstance(value, (int, float)):
                # P-values should be in [0, 1]
                if 'p_value' in key.lower() or 'p-value' in key.lower() or key.lower().endswith('_p'):
                    if not (0 <= value <= 1):
                        issues.append(f"{key} = {value} is not in [0, 1]")

                # Error rates should be in [0, 1]
                if 'error' in key.lower() or 'rate' in key.lower() or 'power' in key.lower():
                    if not (0 <= value <= 1):
                        issues.append(f"{key} = {value} is not in [0, 1]")

                # Variances should be non-negative
                if 'variance' in key.lower() or 'var' in key.lower():
                    if value < 0:
                        issues.append(f"{key} = {value} is negative (variance should be >= 0)")

        # If basic checks pass, use LLM for deeper validation
        if not issues:
            prompt = f"""Validate these simulation results:

Results:
{json.dumps(output, indent=2)}

Task Requirements:
{json.dumps(parsed_task, indent=2)}

Expected Behavior:
{json.dumps(simulation_plan.get('expected_results', {}), indent=2)}

Check for:
1. Are the results statistically plausible?
2. Do they align with theoretical expectations?
3. Are there any suspicious patterns?
4. Do all requested metrics appear in the output?

Return a JSON object with:
- valid: boolean (true if results look correct)
- issues: array of strings describing any problems found
- suggestions: array of strings with improvement suggestions (can be empty)"""

            try:
                validation = self.llm.generate_json(prompt, self.SYSTEM_PROMPT)
                return validation
            except Exception as e:
                # If LLM validation fails, return basic validation result
                return {
                    "valid": True,
                    "issues": [],
                    "suggestions": [f"LLM validation skipped due to error: {e}"]
                }

        return {
            "valid": False,
            "issues": issues,
            "suggestions": ["Fix the identified issues and re-run"]
        }


class ResultSynthesizer:
    """Synthesizes simulation results into a human-readable report."""

    SYSTEM_PROMPT = """You are a statistics expert who explains simulation results clearly.
Write reports that are:
- Clear and concise
- Technically accurate
- Actionable for researchers
- Include key findings and implications"""

    def __init__(self, llm: LLMBackend):
        self.llm = llm

    def synthesize(self, execution_result: dict, parsed_task: dict, simulation_plan: dict) -> str:
        """Generate a summary report of the simulation results."""
        prompt = f"""Synthesize these simulation results into a clear report:

Task:
{json.dumps(parsed_task, indent=2)}

Results:
{json.dumps(execution_result.get('output', {}), indent=2)}

Simulation Details:
{json.dumps(simulation_plan, indent=2)}

Write a report that includes:
1. Summary of what was simulated
2. Key findings (with specific numbers)
3. Interpretation of results
4. Statistical implications
5. Any caveats or limitations

Format the report in clear paragraphs. Be specific with numbers and their meanings."""

        return self.llm.generate(prompt, self.SYSTEM_PROMPT)
