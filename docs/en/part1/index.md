# Part 1: Overview and Infrastructure

## Positioning

Part 1 establishes the shared conceptual framework for the whole book. It explains the objects, boundaries, quality goals, core cost items, and infrastructure layers of large-model data engineering. It is not centered on a single tool or model. Instead, it starts from the data lifecycle and explains why data has become the common constraint behind model capability, cost, and risk control.

In the published structure, this part has three functions. Chapter 1 introduces the background and paradigm shift, explaining why large-model development must move from a model-centered view to joint governance of data and systems. Chapter 2 establishes the quality language used throughout the book, turning noise, duplication, contamination, bias, missingness, and freshness into measurable, reviewable, and enforceable engineering indicators. Chapter 3 maps the quality framework to infrastructure, explaining how ingestion, processing orchestration, storage, indexing, evaluation operations, governance, and security jointly support training and applications.

## Learning Objectives

After reading this part, readers should be able to:

- Explain the key differences between large-model data engineering, traditional data warehouses, and traditional machine-learning data processing.
- Identify the different quality goals of pre-training, instruction tuning, preference alignment, and RAG applications.
- Map common data issues to detectable quality indicators, governance actions, and rollback strategies.
- Draft an initial AI-native data stack and cost-governance plan based on team scale and training goals.

## Chapter Relationships

Chapter 1 introduces the book's core proposition: data quality, data scale, and data diversity jointly define the boundary of model capability. Chapter 2 answers the question "How do we judge whether data is usable?" and provides quality scorecards and governance gates. Chapter 3 answers "What infrastructure carries these governance actions?" and places the later chapters on cleaning, alignment, RAG, DataOps, and compliance into one architecture.

## Full Book Contents

1. [Part 1: Overview and Infrastructure](../part1/index.md)
2. [Part 2: Text Pre-training Data Engineering](../part2/index.md)
3. [Part 3: Multimodal Data Engineering](../part3/index.md)
4. [Part 4: Instruction Fine-tuning and Preference Data](../part4/index.md)
5. [Part 5: Synthetic Data Engineering](../part5/index.md)
6. [Part 6: Reasoning and Agent Data Engineering](../part6/index.md)
7. [Part 7: Application-Level Data Engineering](../part7/index.md)
8. [Part 8: Data Operations and Platform Development](../part8/index.md)
9. [Part 9: Data Assets, Data Products, and Data Contracts](../part9/index.md)
10. [Part 10: Intelligent Data Engineering and Data Engineering Agents](../part10/index.md)
11. [Part 11: Privacy Compliance and Data Security](../part11/index.md)
12. [Part 12: Specialized Datasets and Data Engineering Practice](../part12/index.md)
13. [Part 13: Open-source LLM Data Engineering Recipes and Paradigms](../part13/index.md)
14. [Part 14: Practical Projects](../part14/index.md)

## Part Contents

- [Chapter 1: The Data Revolution in the Era of Large Models](ch01_data_change.md)
- [Chapter 2: LLM Data Lifecycle and Quality Evaluation Framework](ch02_quality_framework.md)
- [Chapter 3: AI-Native Data Stack and Cost Governance](ch03_data_stack.md)

## Suggested Reading Order

- Start with Chapter 1 to build the overall problem awareness that data determines the upper bound of model capability.
- Then read Chapter 2 to master the data lifecycle, quality layers, and evaluation framework.
- Finally read Chapter 3 to ground the framework in platforms, compute, storage, and cost governance.
