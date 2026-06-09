# Part IV: Instruction Fine-Tuning and Preference Data

## Overview

Part IV focuses on supervised data construction for post-alignment models, covering supervised fine-tuning, preference learning, reward signals, annotation platforms, quality assurance, and data operations mechanisms.

## Terminology

Throughout this part, the term **"instruction fine-tuning data (SFT Data)"** refers to input-output samples used for supervised fine-tuning, while **"preference data (Preference Data)"** refers to comparison samples used for ranking, reward modeling, or preference optimization. SFT data, preference data, reward signals, and QA records should each specify their sample objectives, annotation criteria, and acceptance standards separately; avoid grouping all post-training data under the generic label "labeled data."

## Contents

- [Chapter 12: SFT Data Design and Instruction Taxonomy](ch12_sft.md)
- [Chapter 13: Preference Data and Reward Signals](ch13_preference.md)
- [Chapter 14: Annotation Platforms, QA Systems, and Data Operations](ch14_qa.md)

## Recommended Reading Order

- Begin with Chapter 12 to understand SFT sample structure, task design, and instruction templates.
- Proceed to Chapter 13 to learn about preference data, reward modeling, and ranking signals.
- Conclude with Chapter 14 to connect annotation platforms, QA workflows, and organizational operations.
