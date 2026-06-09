# Part X: Intelligent Data Engineering and the Data Engineering Agent

## Positioning of This Part

Part X discusses Agentic Data Engineering, focusing on how data engineering agents participate in collection, parsing, cleaning, annotation, synthesis, evaluation, DataOps, and security collaboration. This part uses DataAgent as a through-line engineering reference: Chapter 31 first situates it within an architecture and boundary framework, while subsequent chapters discuss how collection and cleaning, evaluation, DataOps, and security collaboration each integrate with this agent-based data engineering foundation.

## Terminology

Throughout this part, "data engineering agent" refers to an intelligent component capable of executing data tasks within permission boundaries, invoking tools, recording processes, and supporting audit; "DataAgent" refers to the engineering reference system that runs through this entire part; and "human-machine collaboration" refers to the division of responsibilities among agents, human reviewers, platform policies, and security gates. Agent automation is not equivalent to unattended operation — whenever collection, cleaning, synthesis, evaluation, or security operations are involved, permission scopes, rollback mechanisms, and audit trails must be explicitly specified.

## Table of Contents

- [Chapter 31: Architecture and Task Boundaries of the Data Engineering Agent](ch31_agent_architecture.md)
- [Chapter 32: Automated Collection, Parsing, and Cleaning Agent](ch32_auto_collection_parsing_cleaning.md)
- [Chapter 33: Annotation, Synthesis, and Evaluation Agent](ch33_labeling_synthesis_evaluation.md)
- [Chapter 34: DataOps Agent and Platform Autonomy](ch34_dataops_agent.md)
- [Chapter 35: Security, Permissions, and Human-Machine Collaboration for Data Engineering Agents](ch35_security_permission_collaboration.md)

## Recommended Reading Order

- Begin with Chapter 31 to clarify the architecture, task boundaries, and automation levels of data engineering agents.
- Proceed to Chapters 32 through 34 to understand the core feedback loops of collection and cleaning, annotation-synthesis-evaluation, and DataOps autonomy.
- Conclude with Chapter 35, treating permissions, security, auditing, and human-machine collaboration as the pre-deployment gate.
