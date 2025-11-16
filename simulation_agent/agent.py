"""
Main SimulationAgent class that orchestrates the simulation workflow.
"""

import logging
from typing import Optional
from dataclasses import dataclass, field

from .llm import LLMBackend
from .components import (
    TaskParser,
    SimulationDesigner,
    CodeGenerator,
    ExecutionEngine,
    OutputValidator,
    ResultSynthesizer
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AgentState:
    """Tracks the current state of the agent's workflow."""
    task_description: str = ""
    parsed_task: dict = field(default_factory=dict)
    simulation_plan: dict = field(default_factory=dict)
    generated_code: str = ""
    execution_result: dict = field(default_factory=dict)
    validation_result: dict = field(default_factory=dict)
    final_report: str = ""
    iteration_count: int = 0
    error_history: list = field(default_factory=list)
    code_versions: list = field(default_factory=list)


class SimulationAgent:
    """
    An LLM-powered agent that creates and runs statistical simulations.

    The agent follows this workflow:
    1. Parse the task to understand what simulation is needed
    2. Design the simulation strategy
    3. Generate Python code
    4. Execute the code in a sandboxed environment
    5. Validate the output
    6. Debug and retry if needed
    7. Synthesize and return results
    """

    def __init__(
        self,
        llm: LLMBackend,
        max_retries: int = 5,
        timeout_seconds: int = 300,
        verbose: bool = True
    ):
        """
        Initialize the simulation agent.

        Args:
            llm: LLM backend for reasoning
            max_retries: Maximum number of debug attempts
            timeout_seconds: Timeout for code execution
            verbose: Whether to print progress updates
        """
        self.llm = llm
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.verbose = verbose

        # Initialize components
        self.parser = TaskParser(llm)
        self.designer = SimulationDesigner(llm)
        self.generator = CodeGenerator(llm)
        self.executor = ExecutionEngine(timeout_seconds)
        self.validator = OutputValidator(llm)
        self.synthesizer = ResultSynthesizer(llm)

        # State tracking
        self.state = AgentState()

    def _log(self, message: str, level: str = "info"):
        """Log a message if verbose mode is on."""
        if self.verbose:
            if level == "info":
                logger.info(message)
            elif level == "warning":
                logger.warning(message)
            elif level == "error":
                logger.error(message)

    def run(self, task: str) -> dict:
        """
        Run the full simulation workflow for a given task.

        Args:
            task: Natural language description of the simulation task

        Returns:
            Dictionary containing:
                - success: bool
                - report: str (final report)
                - results: dict (simulation results)
                - code: str (final working code)
                - iterations: int (number of attempts)
        """
        self._log(f"Starting simulation agent for task: {task}")
        self.state = AgentState(task_description=task)

        # Step 1: Parse the task
        self._log("Step 1: Parsing task...")
        try:
            self.state.parsed_task = self.parser.parse(task)
            self._log(f"Parsed task: {self.state.parsed_task}")
        except Exception as e:
            return self._failure_result(f"Failed to parse task: {e}")

        # Step 2: Design the simulation
        self._log("Step 2: Designing simulation...")
        try:
            self.state.simulation_plan = self.designer.design(self.state.parsed_task)
            self._log(f"Simulation plan: {self.state.simulation_plan}")
        except Exception as e:
            return self._failure_result(f"Failed to design simulation: {e}")

        # Step 3-6: Generate, execute, validate, debug loop
        while self.state.iteration_count < self.max_retries:
            self.state.iteration_count += 1
            self._log(f"\n--- Iteration {self.state.iteration_count} ---")

            # Step 3: Generate code
            self._log("Step 3: Generating code...")
            try:
                if self.state.iteration_count == 1:
                    # First attempt - generate from scratch
                    self.state.generated_code = self.generator.generate(
                        self.state.parsed_task,
                        self.state.simulation_plan
                    )
                else:
                    # Subsequent attempts - fix based on errors
                    self.state.generated_code = self.generator.fix(
                        self.state.generated_code,
                        self.state.error_history[-1],
                        self.state.parsed_task,
                        self.state.simulation_plan
                    )
                self.state.code_versions.append(self.state.generated_code)
                self._log(f"Generated code:\n{self.state.generated_code[:500]}...")
            except Exception as e:
                self.state.error_history.append(f"Code generation error: {e}")
                self._log(f"Code generation failed: {e}", "error")
                continue

            # Step 4: Execute code
            self._log("Step 4: Executing code...")
            try:
                self.state.execution_result = self.executor.execute(self.state.generated_code)
                self._log(f"Execution result: {self.state.execution_result}")
            except Exception as e:
                self.state.error_history.append(f"Execution error: {e}")
                self._log(f"Execution failed: {e}", "error")
                continue

            # Check for runtime errors
            if not self.state.execution_result.get("success", False):
                error_msg = self.state.execution_result.get("error", "Unknown execution error")
                self.state.error_history.append(f"Runtime error: {error_msg}")
                self._log(f"Runtime error: {error_msg}", "warning")
                continue

            # Step 5: Validate output
            self._log("Step 5: Validating output...")
            try:
                self.state.validation_result = self.validator.validate(
                    self.state.execution_result,
                    self.state.parsed_task,
                    self.state.simulation_plan
                )
                self._log(f"Validation result: {self.state.validation_result}")
            except Exception as e:
                self.state.error_history.append(f"Validation error: {e}")
                self._log(f"Validation failed: {e}", "error")
                continue

            # Check if valid
            if self.state.validation_result.get("valid", False):
                self._log("Validation passed! Synthesizing results...")
                break
            else:
                issues = self.state.validation_result.get("issues", ["Unknown validation issue"])
                self.state.error_history.append(f"Validation issues: {issues}")
                self._log(f"Validation issues: {issues}", "warning")
                continue
        else:
            # Max retries exceeded
            return self._failure_result(
                f"Max retries ({self.max_retries}) exceeded. Errors: {self.state.error_history}"
            )

        # Step 7: Synthesize results
        self._log("Step 7: Synthesizing results...")
        try:
            self.state.final_report = self.synthesizer.synthesize(
                self.state.execution_result,
                self.state.parsed_task,
                self.state.simulation_plan
            )
        except Exception as e:
            return self._failure_result(f"Failed to synthesize results: {e}")

        return {
            "success": True,
            "report": self.state.final_report,
            "results": self.state.execution_result.get("output", {}),
            "code": self.state.generated_code,
            "iterations": self.state.iteration_count
        }

    def _failure_result(self, message: str) -> dict:
        """Create a failure result dictionary."""
        self._log(message, "error")
        return {
            "success": False,
            "report": message,
            "results": {},
            "code": self.state.generated_code,
            "iterations": self.state.iteration_count,
            "error_history": self.state.error_history
        }
