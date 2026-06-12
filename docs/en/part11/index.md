# Part XI: Privacy Compliance and Data Security

## Overview

Part XI focuses on data compliance, privacy protection, federated learning, and security boundaries. It emphasizes placing regulatory requirements, risk controls, and auditability at the forefront of the data lifecycle within engineering workflows.

## Terminology

This part consistently uses "data compliance" to describe constraints on usage, authorization, retention, deletion, cross-border transfer, and auditing; "privacy-preserving technologies" to describe engineering techniques such as differential privacy, federated learning, cryptographic computation, and data minimization; and "security boundaries" to describe permissions, isolation, access control, and misuse prevention. Compliance requirements should not be treated as appendix-style declarations — they must be embedded as upstream gates in the data lifecycle, model training, and production operations.

## Learning Objectives

After completing this part, readers should be able to:

- Map authorization, purpose, retention, deletion, cross-border transfer, and audit requirements into data engineering workflows.
- Identify the different risks among personal information, sensitive data, business secrets, and model training data.
- Choose privacy-preserving technologies such as federated learning, differential privacy, cryptographic computation, and data minimization.
- Establish compliance gates, human review, evidence retention, and release-check mechanisms for data projects.

## Prerequisites

Before reading this part, readers should understand the data lifecycle from Part 1, platform governance from Part 8, and agent security boundaries from Part 10. Readers from compliance or security teams may focus on how engineering actions produce evidence, rather than only adding declarations at the end of a project.

## Chapter Logic

Chapter 36 discusses data compliance frameworks and governance, focusing on classification, purpose constraints, audit closure, and organizational responsibility. Chapter 37 moves into federated learning and privacy-preserving technologies, explaining how privacy-enhancing methods are embedded in real training and collaboration workflows. Together, the two chapters move compliance from static principles into executable engineering gates.

## Table of Contents

- [Chapter 36: Data Compliance Framework and Governance](ch36_compliance_framework_and_governance.md)
- [Chapter 37: Federated Learning and Privacy-Preserving Technologies](ch37_federated_learning_and_privacy_preserving_technologies.md)

## Recommended Reading Order

- Start with Chapter 36 to understand data classification, usage constraints, audit closure, and governance responsibilities.
- Then read Chapter 37 to learn how federated learning, differential privacy, and privacy-enhancing technologies are implemented in engineering practice.
