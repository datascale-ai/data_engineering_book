# Part IV: Instruction Fine-Tuning and Preference Data

## Overview

Part IV focuses on supervised data construction for post-alignment models, covering supervised fine-tuning, preference learning, reward signals, annotation platforms, quality assurance, and data operations mechanisms.

## Terminology

Throughout this part, the term **"instruction fine-tuning data (SFT Data)"** refers to input-output samples used for supervised fine-tuning, while **"preference data (Preference Data)"** refers to comparison samples used for ranking, reward modeling, or preference optimization. SFT data, preference data, reward signals, and QA records should each specify their sample objectives, annotation criteria, and acceptance standards separately; avoid grouping all post-training data under the generic label "labeled data."

## Learning Objectives

After completing this part, readers should be able to:

- Design an SFT sample system that covers tasks, formats, tone, safety boundaries, and refusal strategies.
- Distinguish the uses of preference data, reward signals, ranking samples, and human QA records.
- Build the collaboration chain among annotation platforms, quality-inspection sampling, dispute arbitration, and data operations.
- Judge whether a batch of post-training data is traceable, reviewable, and sustainable for iteration.

## Prerequisites

Before reading this part, readers should understand text data governance from Part 2 and multimodal sample structures from Part 3. Readers focused on model training may pay special attention to how raw task requirements are converted into stable supervision signals. Readers from annotation or operations teams may focus on sample definitions, acceptance standards, and dispute-handling mechanisms.

## Chapter Logic

Chapter 12 answers what kinds of input-output samples are suitable for supervised fine-tuning, focusing on task systems, instruction templates, and coverage. Chapter 13 answers how preferences and reward signals become trainable data, focusing on pairwise comparisons, ranking, reward modeling, and safety feedback. Chapter 14 answers how annotation and QA become an operable system, focusing on platform workflows, sampling inspection, annotation consistency, and long-term data operations.

## Contents

- [Chapter 12: SFT Data Design and Instruction Taxonomy](ch12_sft.md)
- [Chapter 13: Preference Data and Reward Signals](ch13_preference.md)
- [Chapter 14: Annotation Platforms, QA Systems, and Data Operations](ch14_qa.md)

## Recommended Reading Order

- Begin with Chapter 12 to understand SFT sample structure, task design, and instruction templates.
- Proceed to Chapter 13 to learn about preference data, reward modeling, and ranking signals.
- Conclude with Chapter 14 to connect annotation platforms, QA workflows, and organizational operations.
