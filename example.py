#!/usr/bin/env python3
"""
Example usage of the SimulationAgent.

This script demonstrates how to use the simulation agent to automatically
create and run a statistical simulation.
"""

from simulation_agent import SimulationAgent, AnthropicBackend

def main():
    # Initialize the LLM backend (uses ANTHROPIC_API_KEY environment variable)
    llm = AnthropicBackend()

    # Create the simulation agent
    agent = SimulationAgent(
        llm=llm,
        max_retries=5,
        timeout_seconds=300,
        verbose=True
    )

    # Define a simulation task
    task = """
    Simulate the Type I error rate of a two-sample t-test when the data comes from
    a log-normal distribution instead of a normal distribution.

    Test with sample sizes of 10, 30, and 100 per group.
    Use 10000 simulations for each sample size.
    The null hypothesis is true (both groups have the same distribution).
    Report the empirical Type I error rate at alpha = 0.05.
    """

    # Run the simulation
    print("=" * 60)
    print("SIMULATION AGENT")
    print("=" * 60)
    print(f"\nTask: {task.strip()}\n")
    print("=" * 60)

    result = agent.run(task)

    # Display results
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    if result["success"]:
        print(f"\nStatus: SUCCESS (completed in {result['iterations']} iteration(s))")
        print("\n--- Simulation Results ---")
        for key, value in result["results"].items():
            print(f"  {key}: {value}")

        print("\n--- Final Report ---")
        print(result["report"])

        print("\n--- Generated Code ---")
        print("```python")
        print(result["code"])
        print("```")
    else:
        print(f"\nStatus: FAILED after {result['iterations']} iteration(s)")
        print(f"\nError: {result['report']}")
        if "error_history" in result:
            print("\nError History:")
            for i, error in enumerate(result["error_history"], 1):
                print(f"  {i}. {error}")


if __name__ == "__main__":
    main()
