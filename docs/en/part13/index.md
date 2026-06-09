# Part XIII: Data Engineering Recipes and Paradigms for Open-Source Large Language Models

## Positioning of This Part

Part XIII transitions from specialized case studies to open-source model data recipes, with the goal of abstracting long-term, transferable data engineering paradigms. Open-source model reports, training logs, and companion repositories frequently contain large numbers of model names, dataset names, and benchmark conclusions; the focus of this part is not to chase individual trending models, but to identify the data organization principles underlying different training stages.

This part covers five categories of recipes: pretraining data recipes, post-training data recipes, RL/reasoning data recipes, VLM data recipes, and T2I/T2V generative model data recipes. These correspond respectively to the questions of "what raw materials does the model learn from," "how is controllable behavior acquired," "how is verifiable reasoning formed," "how is vision-language alignment achieved," and "how generative models absorb high-quality image-text/video supervision."

Looking backward, this part builds on the methods, platforms, evaluation frameworks, and compliance foundations established in Parts I through XII. Looking forward, it provides recipe coordinates for the open-source model reproduction, inference flywheels, multimodal instruction factories, video generation pipelines, and DataAgent application cases in Part XIV (P11–P15).

## Terminology Conventions

Throughout this part, "data recipe" is used consistently to describe the combinatorial relationship among data sources, mixing ratios, processing strategies, training stages, and evaluation objectives. "Post-training" is used as an umbrella term covering all model behavior shaping stages, including SFT, preference alignment, RL, and reasoning reinforcement. "VLM data recipe" and "T2I/T2V data pipeline" are used to describe the data organization approaches for understanding-oriented and generation-oriented multimodal models, respectively. When citing open-source technical reports, a clear distinction should be maintained among publicly verifiable conclusions, engineering inferences, and transferable recommendations synthesized by this book.

## Table of Contents for This Part

- [Chapter 44: LLM Pretraining Data Engineering in Practice: From Recipe to Production](ch44_pretrain_recipes.md)
- [Chapter 45: LLM Post-Training Data Engineering in Practice: SFT and Preference Alignment](ch45_posttrain_recipes.md)
- [Chapter 46: Reasoning Models and RL Data Engineering: The R1 / QwQ Paradigm](ch46_rl_reasoning_data.md)
- [Chapter 47: Multimodal Large Model (VLM) Data Recipes: From Pretraining to Visual Alignment](ch47_vlm_data_recipes.md)
- [Chapter 48: Generative Multimodal Model Data Engineering: T2I and T2V Data Pipelines](ch48_t2i_t2v.md)

## Recipe Paradigms

Pretraining recipes focus on corpus sources, mixing ratios, deduplication, quality filtering, token distributions, and training sample packing. The central question is how to establish an interpretable balance among scale, quality, diversity, and cost.

Post-training recipes focus on SFT, preference data, rejection sampling, instruction coverage, safety boundaries, and evaluation feedback loops. The central question is how to convert general model capabilities into controlled behaviors while keeping sample quality traceable.

RL/reasoning recipes focus on long-chain reasoning, process supervision, verifiers, reward signals, rollback data, and failure trajectories. The central question is how to extend reasoning correctness from final answers to process structure.

VLM recipes focus on image-text alignment, OCR, visual question answering, document images, chart understanding, and multi-turn dialogue. The central question is how to maintain consistency among visual evidence, textual instructions, and training objectives.

T2I/T2V recipes focus on images, video, captions, motion information, quality scores, safety filtering, and copyright boundaries. The central question is how to transform media assets into trainable, auditable, and releasable supervision data for generative models.

## Recommended Reading Order

It is recommended to read Chapters 44 and 45 first to establish baseline data recipes for pretraining and post-training, then proceed to Chapter 46 to understand the divergence between reasoning models and RL data engineering, and finally read Chapters 47 and 48 to enter the data recipes for VLMs and generative multimodal models.

For further reproduction work, readers may proceed to Part XIV (P11–P15) and cross-reference the recipes abstracted in this part against the resource constraints, acceptance metrics, and failure patterns found in the project case studies.
