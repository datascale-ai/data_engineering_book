# Part VI: Reasoning and Agent Data Engineering

## Positioning of This Part

Part VI focuses on reasoning trajectories, tool use, function calling, agent memory, and multi-turn interaction samples. It discusses how to transform complex reasoning and action processes into trainable, verifiable, and reviewable data assets.

## Terminology Conventions

Throughout this part, "reasoning data" refers to samples that carry problem-solving processes, verification paths, or intermediate states; "tool-use data" refers to records containing tool schemas, invocation parameters, execution results, and error recovery; "agent interaction data" refers to trajectories with memory, state, and multi-turn actions. CoT, function calling, and agent memory should each have their supervision targets defined separately, to avoid overgeneralizing all intermediate processes as "chain-of-thought."

## Table of Contents for This Part

- [Chapter 18: Chain-of-Thought and Reasoning Data Engineering](ch18_cot.md)
- [Chapter 19: Tool-Use and Function Calling Data](ch19_tool.md)
- [Chapter 20: Agent Memory and Multi-Turn Interaction Data](ch20_agent.md)

## Recommended Reading Order

- Start with Chapter 18 to understand sample design for CoT and reasoning trajectories.
- Then read Chapter 19 to master tool invocation, function signatures, and execution constraints.
- Finally, read Chapter 20 to extend into agent memory, multi-turn state, and long-horizon interaction data.
