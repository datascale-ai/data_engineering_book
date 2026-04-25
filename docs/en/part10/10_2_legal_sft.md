# Project 2: Vertical Field Expert SFT (Legal)


## Chapter overview

P02 focuses on organizing regulatory texts, system descriptions, and legal task requirements into a trainable, quality-inspectable, and scalable vertical field SFT data production line. The focus of this chapter is not on a single question and answer generation, but on the stable transformation process from seed knowledge to supervisory assets.

This chapter can be understood according to four main lines:

* Seed knowledge processing: Extract usable structured knowledge fragments from regulatory PDFs and institutional texts.
* Task system and sample synthesis: split into different task layers such as legal interpretation, legal Q&A, case analysis, and risk rejection.
* Quality control and preference enhancement: Stabilizing supervisory signals via QA, preference pairs, and risk bounding samples.
* Training packaging and online acceptance: Organize the processed data assets into trainable, verifiable, and deliverable finished products.

If read in engineering order, this chapter corresponds to a complete link:

**Original regulations PDF -> Cleaning and dicing -> Task design -> Instruction synthesis -> Preference enhancement -> QA quality inspection -> Training packaging -> Online acceptance**

The core goal corresponding to this structure is to process legal knowledge into supervisory data assets with task stratification, quality constraints and acceptance mechanisms.

---

## 1. Project Background: The Necessity of Legal SFT Data Factory

General-purpose large models already have very good language expression capabilities in open-domain question answering, but once they enter legal scenarios, problems will be quickly exposed.

There are three most common types of distortion.

The first category is **Knowledge Distortion**. Models will mix similar provisions together, mix old laws with new laws, or turn rules that originally apply only to specific subjects and specific premises into general conclusions. This type of error may just be an "inaccurate answer" in ordinary encyclopedia Q&A, but in a legal scenario, it will directly affect the user's judgment.

The second category is **task distortion**. Many legal answers are not just about "giving a conclusion", but also require the model to identify the issues in the case, distinguish between facts and norms, give applicable conditions, and clearly retain boundaries when uncertain. A model that only memorizes legal regulations does not mean that it is a model that can provide compliance assistance.

The third category is **style distortion**. Legal scenarios have strong requirements on expression style: neither can one jump to random conclusions, nor can one answer all questions as "it is recommended to consult a professional lawyer"; one must be as clear and easy to understand as possible, while retaining the necessary prudent expression. Behind this is actually a behavioral style problem determined by SFT and preferences.

Therefore, the goal of P02 is not to simply "generate some legal questions and answers", but to build a **SFT data factory** in the legal field** to organize regulatory texts, task systems, quality control, preference signals and risk boundaries into a reusable production line.

This production line serves not a one-time experiment, but a methodology:

> When the team needs to migrate from law to taxation, finance, medical care, customer service and other fields in the future, what can really be reused is not a certain prompt, but this set of engineering methods "from seed knowledge to supervision data".

---

## 2. Project goals and boundaries

### 2.1 Project goals

This project focuses on the following four goals.

**Goal 1: Establish a transformation link from seed corpus in the legal field to supervision data. **
That is, the regulatory provisions, system descriptions and related knowledge fragments in unstructured PDFs are converted into structured samples suitable for training.

**Goal 2: Establish a task system oriented to legal scenarios. **
This project does not uniformly make all samples into "question and answer pairs", but clearly splits them into different task types such as legal question and answer, interpretation of legal provisions, and case analysis, so that the model can learn different forms of domain capabilities.

**Goal 3: Establish an auditable, rejectable, and versionable QA mechanism. **
If legal data is only generated without review, it is easy to amplify erroneous samples in batches. Therefore, the project not only has SFT samples, but also builds preference pairs, review records, and risk rejection samples.

**Goal 4: Form data assets that can be directly consumed by the training side. **
The final output includes not only the original intermediate products, but also training interface layer assets such as `train.jsonl`, `val.jsonl`, `smoke_test.jsonl`, `training_manifest.json`, etc.

### 2.2 Project Boundaries

In order to keep the project reproducible and have clear boundaries, this project explicitly sets several boundaries.

#### 1) Boundaries of knowledge sources

The current scope is mainly Chinese legal texts, mainly from regulatory and institutional texts, rather than massive real user consultation records, a full database of judgment documents or lawyers' working papers. This means that this project is more suitable as a method demonstration and factory prototype, rather than directly claiming to cover all legal issues in the real world.

#### 2) Task boundaries

This project currently focuses on three types of tasks:

* Legal Q&A (legal_qa)
* Statute_explanation
* Case analysis (case_analysis)

These three types of tasks are enough to cover the main path of "knowledge expression - normative interpretation - fact classification", but they have not yet been fully extended to more complex tasks such as contract review, litigation strategy, search-based citation, and multi-round case handling assistance.

#### 3) Boundaries of supervision methods

Although this project has introduced preference pairs and review records, the overall approach is still based on a hybrid approach of templated teachers + heuristic referees + manual QA**, rather than relying entirely on open-ended human experts writing each item.

#### 4) Boundary of online capabilities

Risk rejection samples and risk registration have been included, but the sample size is still small and is suitable for demonstrating how to introduce safety boundaries in the factory, but should not be exaggerated to say that it is "enough to support production online."

### 2.3 The role of boundary setting

It is very important to write clear boundaries. Because there are usually only two ways to expand an engineering project:

* One is to write the project so that it “can do anything”;
* The other is to write the project as "what can be done stably and under what conditions".

The latter is obviously more credible and more suitable for reuse by the team.

---

## 3. Project positioning: P02’s capability chain position

If the whole book is regarded as a large model data engineering capability chain, then P02 is at the core of the section "Instruction fine-tuning and preference data".

Previous chapters have discussed methodologies such as general SFT data design, preference data and reward signals, annotation platforms and QA systems. The value of this chapter is to bring these methods back to a **real industry scenario**: law.

In other words, this chapter is not to re-teach the general knowledge of SFT, but to show:

* In a highly professional, high-risk, and strong compliance scenario, what new problems will SFT data design encounter?
* Why splitting legal tasks can’t just copy the universal Q&A template;
* Why QA must be front-loaded into the production process here;
* Why SFT alone is not enough, we also need to build preference pairs and risk rejection samples;
* How to take version evolution, cost and human-machine collaboration into consideration at the early stage of the project.

In this sense, the most important thing about this chapter is not a "list of technical components" but answering a larger question:

> How should the industry SFT data factory be designed as a set of continuous production capabilities instead of a one-time data synthesis script?

---

## 4. Overall architecture: Legal data pipeline from regulatory PDF to training assets



![图 1：法律领域 SFT 数据工厂总览](../../images/part10/10_2_fig01_legal_sft_factory_overview.png)

From an engineering perspective, this project can be broken down into three floors.

### 4.1 The first layer: knowledge processing layer

This layer addresses "whether there are clean and controllable pieces of legal knowledge." Mainly include:

* PDF parsing
* Header and footer cropping
* Chinese word segmentation repair
* Embedded page number cleaning
* Clause slicing and structuring

The goal of this step is not to generate training samples, but to convert the original legal text into knowledge units suitable as supervision seeds.

### 4.2 The second layer: supervision structure layer

This layer solves the problem of "how to convert knowledge fragments into different types of training samples." Mainly include:

* Task type division
* Prompt template and command system design
* Self-Instruct synthesis
* CoT explicit
* preference pair construction
* Risk rejection sample structure

This step is the core part of the entire project, because it determines whether the model learns to "memorize the provisions" or "work stably in legal tasks."

### 4.3 The third layer: quality inspection and delivery layer

What this layer solves is "whether these samples can really be used for training and online." Mainly include:

* QA audit records
* Accept/Reject Rules
* training split
* manifest generation
* evaluation report
* Project check script

At this point, the project has changed from "data generation experiment" to "engineering closed loop".

---

## 5. Pre-engineering: key aspects of legal data factory

When many teams are new to SFT in the vertical field, by default, one algorithm engineer will be responsible for knowledge collation, template writing, quality review and training set packaging. But in legal settings, such a mix of roles often breaks down quickly.

A more reasonable split usually includes at least the following types of roles.

### 5.1 Domain design and knowledge boundaries

Responsible for defining task boundaries, determining sample types, sorting out legal coverage, and identifying high-risk issues. This role does not necessarily have to be a practicing lawyer, but at least he must be able to distinguish which questions are "answerable knowledge questions and answers" and which questions are approaching "individual legal opinions."

### 5.2 Data processing and structured orchestration

Responsible for PDF parsing, cleaning rules, slicing logic, data schema, intermediate product placement, segmentation and inspection. What this role cares about is the stable production capacity of data, not how beautifully written a single answer is.

### 5.3 Generation control and task orchestration

Responsible for Self-Instruct templates, task sampling, prompt word arrangement, result post-processing, batch calls and failure retries. It connects "knowledge input" and "supervised sample output".

### 5.4 QA and acceptance closed loop

Responsible for formulating audit protocols, random inspection rules, rework mechanisms, error labels and upgrade paths. In legal scenarios, this role is very critical, because whether the project is ultimately usable does not depend on how many samples the model has generated, but on whether erroneous samples have been identified and kept out.

### 5.5 The role of key responsibility areas

Because when many teams are doing industry SFT for the first time, what is really stuck is not "not being able to write code", but the failure to split the production process into roles and links, resulting in:

* Task boundaries are not defined
* No one is responsible for whether the sample is correct or not.
* Rework rules are not implemented
* Version updates rely entirely on verbal synchronization

Clearly writing down the division of responsibilities is essentially explaining that: **Industry SFT is more like a content production line than a single-point script. **

![图 2：法律 SFT 数据工厂角色分工图](../../images/part10/10_2_fig02_roles_and_responsibilities.png)

---

## 6. Seed data: The seed layer serves as the starting point for supervision

In general Q&A, many teams will directly grab question and answer pairs from knowledge bases, web pages, or forums as training data. But the legal scene is not suitable for this.

The reason is simple: in legal Q&A, user statements are often incomplete and the sources are not necessarily authoritative. If you directly use open question and answer as training ground truth, the model will learn a lot of vague expressions, unverified conclusions and unstable styles.

Therefore, this project starts with legal and institutional texts to build a relatively stable seed corpus. The value of this layer is not to "cover all user problems", but to provide a traceable, explainable, and dicable knowledge base.

### 6.1 Regulatory texts as first seeds

* The structure is relatively clear and suitable for cutting into pieces;
* Relatively high authority and suitable for basic supervision;
* Facilitates explanation of practice instructions, knowledge Q&A and summary of rules;
* Quality control is easier to accomplish on small-scale projects.

### 6.2 Boundaries of regulatory text

Data factories also encounter two obvious problems if they rely solely on regulatory text.

First, regulatory texts are naturally biased toward “standardized expressions” and are not equivalent to the actual way users ask questions.
Second, regulatory text is more suitable for supporting "explanation" and "citation", but it does not provide sufficient coverage for complex case analysis, vague fact classification, and colloquial expression of business scenarios.

Therefore, regulatory text is suitable as a first-level seed, but should not be mistaken for complete surveillance data itself.

---

## 7. PDF parsing and intelligent cleaning: format cleaning of legal texts

One of the biggest features of laws and regulations PDF is that the content is very rigorous, but the format is extremely unfriendly to machines.

When humans read, headers, footers, page numbers, watermarks, paragraph breaks, double-column layouts, and word breaks do not cause much trouble; but for machines, these are sources of noise that will contaminate training samples.

### 7.1 Limitations of plain text extraction

When many beginners process PDFs, they will directly use tools that can output strings, and then send the text for segmentation. This approach may be barely usable in ordinary scenarios, but it is very problematic in legal scenarios.

Because once the legal text is parsed and misaligned, it is most likely to cause two disasters:

* Clauses that were originally separate were spelled into one sentence;
* The originally continuous logic of legal articles was fragmented by page numbers, headers, and word breaks.

This will not only affect the readability of the sample, but also allow subsequent Self-Instruct to generate supervision data based on the wrong fragments that "looks reasonable, but in fact the source is wrong."

### 7.2 Component selection

| Components | Selection | Function | Reason for selection |
| ------ | ------------- | ------------- | ------------------------------------ |
| PDF parsing | `pdfplumber` | Read page text and coordinates | Can do header and footer cropping based on Bounding Box, suitable for processing institutional PDFs |
| Cleaning logic | `Regex` | Repair word segmentation, remove page numbers, and remove dirty characters | Many errors in legal PDFs are rule-based, and regularization is the most directly controllable in the early stages |
| Generative model | `DeepSeek-V3` | Instruction synthesis and inference expansion | Taking into account the quality and cost of inference, suitable for large-scale synthesis |
| Arrangement logic | `Python` | Batch processing, sampling, post-processing | Facilitate the rapid construction of the smallest reproducible process |

### 7.3 Crop header and footer

The most typical repetitive noise in legal PDFs comes from headers and footers. For example, the name of the regulation is repeated at the top of each page, and the page number or publication information is repeated at the footer. If they are not removed during the parsing stage, they will be mistaken for text and repeatedly entered into the training data.

Therefore, when this project reads each page, it directly cuts off about 5% of the upper and lower areas, leaving only the middle text area. The advantages of doing this are:

* It is more stable than cleaning after extraction, because the noise from the source is reduced;
* Good adaptability to most regulatory PDFs;
* The implementation is simple and suitable as a minimum reproducible solution.

The corresponding implementation is as follows:

```python
with pdfplumber.open(file_path) as pdf:
    for page in pdf.pages:
        width, height = page.width, page.height
        bbox = (0, height * 0.05, width, height * 0.95)
        page_crop = page.crop(bbox=bbox)
        text = page_crop.extract_text()
```

### 7.4 Remove embedded page numbers

Compared with headers and footers, a more subtle problem is the page number embedded in the text, for example:

```text
……应当依法承担相应责任。 - 195 - 当事人……
```

If you simply write a "delete dash number dash" rule, it is easy to accidentally delete the numbering or list structure that legally exists in the text. Therefore, this project constrains the front and rear boundaries of page numbers through more careful regularization, and only deletes fragments that are more like independent page number blocks without easily touching the text number.

This kind of cleaning looks like "glue code", but it is often very critical in engineering. Because what it determines is:

> Are we making subtle repairs to the legal text, or are we using crude rules to undermine the text itself?

### 7.5 Repair Chinese word segmentation

Another common problem with Chinese PDFs is "false spaces", such as:

```text
法 律 规 定
合 同 关 系
```

For humans, this does not affect reading, but for the model, it will destroy word segmentation statistics, affect generation fluency, and reduce the availability of downstream samples. Therefore, this project makes regular repairs to abnormal spaces between adjacent Chinese characters, and handles the situation of continuous word breakage through multiple replacements.

### 7.6 The necessity of fine-grained cleaning control

Because the first step of industry SFT is never to "find ways to generate more data", but to ensure that the seed layer is not dirty first. As long as there is a lot of formatting damage in the seed text, subsequent templates, CoT, preferences and QA will all work on a dirty basis, and the cost will only get higher and higher.

![图 3：法律 PDF 智能清洗示意图](../../images/part10/10_2_fig03_pdf_cleaning_pipeline.png)

![图 4：嵌入式页码与中文断词修复案例](../../images/part10/10_2_fig04_cleaning_examples.png)

---

## 8. Dicing and schema: A structured approach to legal seeds

After completing the basic cleaning, the project will not directly send the long text into the generation model, but will first do **cutting and structuring**.

### 8.1 Cutting into pieces as a necessary step

A regulatory or institutional text is often very long, and there are three problems with directly inputting it into the model:

* The context is too long, expensive and noisy;
* The topics in different articles are mixed and it is not suitable to be used as a single supervision unit;
* It is difficult to trace the source of the sample in the future, which is not conducive to QA and backtracking.

Therefore, a more reasonable approach is to cut each piece into pieces based on legal provisions, paragraphs of provisions, or relatively independent pieces of knowledge, and turn each piece into a traceable seed sample.

### 8.2 What problem does schema solve here?

The so-called schema is not to look good, but to enable all subsequent links to collaborate based on unified fields. A typical legal seed sample should at least contain:

* `id`: unique identifier
* `source_name`: Source regulation or system name
* `article_no`: bar number or chapter position
* `text`: Cleaned text fragment
* `task_type`: What type of tasks will be expanded into in the future?
* `risk_level`: Whether it is a high-risk topic
* `metadata`: Additional information such as version, cleaning log, analysis source, etc.

With this layer of schema, the project can follow:

* Track which law a certain supervision sample comes from;
* Compare sample distributions from different sources;
* Separate triage of high-risk samples;
* After discovering problems in the training set, check the upstream seed.

### 8.3 schema as seed layer base

Because many data projects fail, not because of poor models, but because unified fields were not established at the beginning, resulting in:

* QA cannot review sources;
* Preference pairs cannot be associated with subject samples;
* train/val segmentation cannot be isolated by source;
* There is no way to start version rollback.

The schema is the foundation of the industry's SFT factory, not an appendage.

![图 5：法律种子样本 schema 示意图](../../images/part10/10_2_fig05_seed_schema.png)

---

## 9. Task system: Task hierarchy of legal SFT

If you ask a team to do legal SFT intuitively, the easiest data set to obtain is often like this:

> Instruction: Please explain Article XXX of the Civil Code.
> Output: This article stipulates...

Such samples are certainly valuable, but if the entire data set looks like this, the model will end up just being a "law repeater."

What the legal scene really requires is not just reciting normative texts, but the ability to work stably in different task forms. Therefore, this project breaks down the supervision tasks into at least three major categories.

### 9.1 Legal Q&A (legal_qa)

This type of task is oriented towards scenarios that are closer to real users asking questions. The emphasis is on:

* Translate normative expressions into questions that users can understand;
* Give answers in relatively clear natural language;
* Provide conditional explanations and boundary hints when necessary.

This type of task trains the "user interface capabilities" of the model.

### 9.2 Statute_explanation

This type of task is geared toward textual understanding and normative interpretation, emphasizing:

* Restoring the meaning of the provisions;
* Explain applicable conditions;
* distinguish key concepts;
* If necessary, indicate which situations are not directly covered by this article.

This type of task trains the "standard expression ability" of the model.

### 9.3 Case analysis (case_analysis)

This is the type of task closest to legal reasoning, with emphasis on:

* Distill arguments from facts;
* Determine how relevant legal provisions may apply;
* Explain the conditions and uncertainties for the conclusion to be established;
* Avoid making arbitrary conclusions when the facts are insufficient.

This type of task trains the model’s “fact-rule mapping ability”.

### 9.4 The quality control role of task splitting

Task splitting is not to make the table more beautiful, but to avoid a very common problem:

> It seems like there are a lot of samples, but in fact all the samples are training the same ability repeatedly.

In the legal field, this problem of “diversity in appearance but unity in reality” is particularly serious. Only by clearly distinguishing between question-answering, explanation, analysis and other capabilities can the model have the opportunity to learn a more complete behavior distribution.

![图 6：法律任务体系分层图](../../images/part10/10_2_fig06_task_taxonomy.png)

---

## 10. Task distribution and sample structure: distribution balanced control method

With the task system in place, the next question is not "can it be generated?" but "whether the generated distribution is healthy."

In the current products of this project, there are 2,577 tasks in each of the three main categories, indicating that the task structure remains relatively balanced. The final training set size was 7,737 entries, indicating that the project has been able to expand from regulatory seeds to a more complete set of supervisory data assets. At the same time, the distribution of legal sources is uneven: there are 3882 relevant samples of the Civil Code, 1710 articles of the Criminal Law, and 855 articles of the Company Law. There is still obvious concentration in the coverage of different legal areas.

This set of numbers means at least three things:

First, the project is no longer "randomly making some samples", but has formed a relatively clear task structure.
Second, the factory already has the ability to scale from seeds to multi-task supervision samples.
Third, legal coverage is still one of the areas that needs to be optimized most in the next step.

### 10.1 Task balancing and source distribution

If we only look at task types, the three types of samples are indeed balanced. But if you continue to look down at the sources, you will find that the samples are mainly concentrated in a few legal jurisdictions. This means that although the model is relatively balanced in terms of "task form", it may still be biased towards certain areas in terms of "knowledge distribution".

### 10.2 The importance of sample structure

Because the total number of samples can only answer "how big is the scale", but cannot answer "where will the model be biased?" In industry data engineering, distribution structure is often more important than absolute size.

![图 7：任务分布与法域覆盖对照图](../../images/part10/10_2_fig07_task_vs_domain_distribution.png)

---

## 11. Self-Instruct: The necessity of controlled synthesis

From regulatory seeds to SFT samples, the most critical step in the middle is synthetic expansion. The project does not require humans to write all legal questions and answers one by one. Instead, the project uses the Self-Instruct method to allow the teacher model to automatically generate candidate samples based on legal provisions and task templates.

### 11.1 The role of synthetic expansion

If you rely solely on manual writing, the project will quickly be overwhelmed by costs. Based on the estimated manual review hours of the current project, it has reached 193.28 hours, and the corresponding review cost is approximately 23,193.6 yuan. If the main sample is all written manually, the overall investment will be higher. On the contrary, it is more realistic to first use the teacher model to automatically expand, and then focus manpower on review and difficult cases.

### 11.2 Constraints on legal composition

In general Q&A, generative models can write a variety of natural language answers relatively freely; but in legal scenarios, the greater the freedom, the higher the risk. Because the model is easy:

* Splice irrelevant clauses into seemingly correct answers;
* Use common sense to supplement conclusions that are actually unfounded;
* Overconfidently outputting “what should be done” suggestions;
* Leave no hint of uncertainty on boundary issues.

Therefore, this project uses synthesis under template constraints rather than completely open generation. The degree of freedom of the teacher model is controlled by the task template and format requirements.

### 11.3 Weighted Roulette and Task Sampling

In order to make the data distribution meet expectations, the project does not do a simple average randomization of the three types of tasks, but samples through a weighted roulette mechanism. Its core idea is:

* Complex case analysis can better train the model's high-value reasoning ability, so it has a higher weight;
* Tasks such as legal documents or concept analysis are also important, but they do not need to overcrowd the sample quota at this stage;
* Task ratio should be an explicitly adjustable engineering parameter, rather than a black box implicit in random numbers.

The value of this approach is that it turns "data distribution" into a controllable object rather than a post-statistical result.

![图 8：加权轮盘赌任务采样示意图](../../images/part10/10_2_fig08_weighted_task_sampling.png)

---

## 12. CoT Explicitness: Expressive Constraints on Legal Reasoning

When many teams do legal SFT, it is easy to interpret "expert feeling" as "long and formal answers". But what really makes a model more like a legal assistant is often not the length, but whether the reasoning process is visible, whether the conclusions are hierarchical, and whether the boundaries are expressed**.

### 12.1 The role of explicit thinking chain

Case analysis tasks particularly rely on intermediate reasoning. If the training data only retains the final conclusion, the model will often only learn to output the result, but will not learn:

* Identify the points of contention first;
* Then determine the applicable standards;
* Then give a conditional conclusion;
* Finally, there are uncertainties.

Therefore, during the post-generation processing of the project, we will try our best to make the "thinking process" field in the model output explicit, and then splice it into a unified Markdown or segmented format. The purpose of this is not to pursue "the model looks like thinking", but to provide a more complete behavioral template for the training process.

### 12.2 Legal CoT usage boundaries

CoT in legal scenarios cannot be infinitely expanded. Reasoning that is too long, too detailed, or too similar to an internal derivation log may not necessarily be suitable for answering directly as an end user. A more practical approach is to control CoT at the "structured reasoning" level, for example:

1. Refining the points of contention
2. comparison rules
3. Analyze applicable conditions
4. Give conclusions and boundaries

This form preserves the reasoning path without turning the sample into lengthy self-talk.

### 12.3 Engineering value of CoT

In this project, the value of CoT is mainly reflected in two aspects:

* Help the model learn an expression order that is closer to legal analysis;
* Provide a clearer intermediate basis for QA review, making it easier to identify samples with "right conclusions but wrong reasoning".

![图 9：案情分析类 CoT 结构示意图](../../images/part10/10_2_fig09_cot_structure.png)

---

## 13. Preference pairs and review records: multi-layer supervision signals

Only SFT samples can often only tell the model "what is an acceptable answer"; but in legal scenarios, this is not enough. Because many answers are not simply “right” or “wrong” but involve differences in style, risk, boundary control, and caution in expression.

This is where preference pairs are important.

### 13.1 The role of preference pairs in legal scenarios

It solves the following situation:

* Both answers are basically correct, but one is more restrained, clearer, and less arbitrary;
* Both answers cite rules, but one is more descriptive of the conditions under which it applies;
* Both answers gave advice, but one better differentiated between informative notes and legal advice.

These differences are difficult to express with a single label, and preference alignment is suitable for expressing "which one is better".

### 13.2 Preference signal construction for current projects

The current products of the project show that the number of preference pairs is 7731, which is basically parallel to the number of SFT accepted by the subject. This shows that the factory does not complete the SFT first and then temporarily add some preferences, but treats the preference signal as an asset to be constructed in parallel with the main supervision from the beginning.

### 13.3 The role of review records

After many teams complete QA, they only keep "passed" samples and do not keep review records. This leads to two problems:

* No one later knows why this sample was accepted or rejected;
* Unable to recycle failed samples and reversely revise templates.

Therefore, this project includes review records as part of the product. The benefits of doing this are:

* Traceable sample quality history;
* Can support secondary arbitration;
* Error patterns can be precipitated from rejection reasons;
* This can provide a basis for the next round of template optimization.

![图 10：偏好对与评审记录关系图](../../images/part10/10_2_fig10_preference_and_review.png)

---

## 14. Risk Denial: Border Control Data

Legal scenarios are typically high-stakes scenarios. The model is not "the more answers, the better", but it must know when to reject, transfer to manual, or keep the boundary.

### 14.1 The relationship between risk rejection and system prompt

Many teams will think before going online: Anyway, the system prompt has already written "Do not provide specific legal advice", and that is enough. In fact, it's far from enough.

Because the behavioral patterns that the model actually learns first come from the training data. If there are a lot of them in the training set:

* Draw direct conclusions about individual cases;
* Make conclusive recommendations on issues where evidence is insufficient;
* Output overly operational answers to highly sensitive questions;

Then it is often difficult to suppress these behaviors stably by just relying on the system prompt during reasoning.

### 14.2 The role of risk rejection samples

The essence of risky refusal samples is to provide the model with a behavioral paradigm of "how to safely not answer". For example:

* Clearly state the lack of information;
* Reminders need to be combined with specific facts and evidence;
* Distinguish between general legal information and individual case opinions;
* It is recommended to seek further judgment from professionals.

### 14.3 Risk boundary construction for current projects

In the existing products, there are 6 risk rejection samples and risk registration items. This is a small amount, but it sends an important signal: the project has moved the risk boundary from a "verbal consideration" to an explicit data asset.

![图 11：法律场景风险拒答分流图](../../images/part10/10_2_fig11_risk_refusal_flow.png)

---

## 15. QA Protocol: Quality Gate for Legal Data

The most underestimated aspect of industry SFT is QA. Many times, teams focus most of their energy on "how to generate more samples", but ignore "how to block bad samples" that really determines the quality of the online process.

In a legal scenario, a qualified QA protocol should answer at least three things:

1. What kind of samples should be accepted?
2. What samples must be rejected?
3. How do you rework after a problem is discovered instead of just doing a one-time cleaning?

### 15.1 Audit dimensions

The audit dimensions can be split into five items:

* **Correctness**: Whether the conclusion is consistent with seed regulations and mission intentions;
* **Completeness**: Whether key conditions, exceptions or applicable prerequisites are missing;
* **Clarity of expression**: Whether it can be understood by non-expert users;
* **Format consistency**: Whether it conforms to the established output template;
* **Risk Boundary**: Whether it crosses the boundary to give case-by-case, arbitrary or highly sensitive advice.

### 15.2 Receiving, Rework and Rejection Rules

An executable QA protocol should not only have two states: "pass/fail". A more practical design would at least include:

* **Accept**: can directly enter the training set;
* **Revise**: The main body is correct, but the expression, format or boundaries are insufficient and need to be reworked;
* **Reject**: Factual or specification errors, risk too high, task mismatch, not entered into training assets.

### 15.3 Error labeling

It is recommended that rejected samples be incorrectly labeled in the QA record, for example:

* Citation error
* The conclusion is out of bounds
* Condition missing
* Improper style
* task mismatch
* Forced answer despite insufficient facts

### 15.4 Necessity of QA protocol

If you only write generation logic and do not write QA protocols, industry SFT will easily degenerate into "data generation tutorials" instead of "data factory methods".

![图 12：QA 审核闭环图](../../images/part10/10_2_fig12_qa_loop.png)

![图 13：QA 接收/返工/拒收判定表](../../images/part10/10_2_fig13_qa_decision_table.png)

---

## 16. Supplier collaboration and human-machine division of labor: audit mechanism under scale expansion

In small-scale experiments, team members can often complete most of the review work themselves. But once the project enters continuous iteration, the cost of manual review will quickly become a major bottleneck.

The estimated manual review hours for the current project are approximately 193.28 hours, and the review cost is approximately 23,193.6 yuan. This number shows that even if it is still a method demonstration project, the cost of human review is not low. If we continue to expand without a more reasonable hierarchical review and supplier collaboration mechanism, costs will quickly get out of control.

### 16.1 Hierarchical review

Not all samples require the same level of human intervention. A more reasonable approach is usually:

* Samples with low risks and clear rules will be pre-examined by machine first;
* Medium-risk samples are reviewed by annotators or domain operations;
* High-risk or divergent samples are then escalated to senior personnel or expert arbitration.

### 16.2 Risks of supplier collaboration

Once external annotation or review resources are introduced, teams are most likely to fall into two pits:

* Only give "what to do", not "why to do it";
* Only give specifications, not counterexamples and boundary examples.

This is especially true in legal scenarios. A simple “please judge whether the answer is correct” guideline is not enough. Reviewers need to see:

* Which answers are basically correct, but still need to be returned because they are too arbitrary;
* Even if the answers are conservative, they are not qualified if they are overly conservative;
* Which questions trigger risk rejection rather than continuing to add answers.

### 16.3 Project location of collaboration mechanism

Because the word "factory" in data factory must ultimately fall on the collaboration mechanism. Only writing models, templates and scripts without writing about people and processes will make it difficult to truly support the implementation of the team.

![图 14：人机协同与供应商分层审核图](../../images/part10/10_2_fig14_human_in_the_loop.png)

---

## 17. Training encapsulation: from supervision samples to training interface

After the generation, review and preference structure are completed, the data factory also needs to encapsulate these products into interfaces that can be directly consumed by the training side.

### 17.1 Training encapsulation as a separate stage

Many projects end after completing the samples, but only when they actually enter training do they discover:

* The fields are not unified and the training script cannot be read;
* train/val segmentation is unstable;
* The smoke test cannot represent the real sample format;
* Reports, metrics, and data files don't match each other.

Therefore, training encapsulation is not simply exporting a JSONL, but ensuring that:

* The subject fields are complete;
* Training and verification splits are reproducible;
* The manifest can describe the data range and version;
* Smoke test can quickly expose interface problems.

### 17.2 Main training products of this project

* `final_sft_dataset.jsonl`
* `train.jsonl`
* `val.jsonl`
* `smoke_test.jsonl`
* `training_manifest.json`

### 17.3 The function of smoke test

The value of smoke test is not to evaluate model performance, but to detect obvious problems in the training link as early as possible, such as missing fields, coding errors, inconsistent sample formats, or mismatch between reading logic and manifest.

![图 15：训练封装与交付接口图](../../images/part10/10_2_fig15_training_artifacts.png)

---

## 18. Results display: Overview of current project outputs

Judging from the current results, P02 has formed a relatively complete supervision asset in the legal field, and after adding downstream verification, this project is no longer just a "data factory process run-through", but has begun to have the ability to do lightweight verification of the validity of supervision signals.

### 18.1 Sample size

* Number of seed laws: `2577`
* High value SFT generated by template teacher: `7731`
* Low-quality controls filtered or constructed by heuristic referees: `7731`
* Number of preference pairs: `7731`
* Final training set size: `7737`

This shows that the project has expanded from the original regulatory seed to a sizable and fully structured set of supervision assets, rather than stopping at a small number of manual samples or one-off sample data.

### 18.2 Task distribution

The current three main categories of tasks remain fully aligned:

* `legal_qa = 2577`
* `statute_explanation = 2577`
* `case_analysis = 2577`

This distribution means that the project has good matching control capabilities at the task level, and there is no situation where a certain task type significantly occupies the sample space.

### 18.3 Source distribution

The current distribution of legal sources is:

* `中华人民共和国民法典 = 3882`
* `中华人民共和国刑法 = 1710`
* `中华人民共和国民事诉讼法 = 951`
* `中华人民共和国公司法 = 855`
* `中华人民共和国劳动法 = 333`

This shows that the project already has the ability to expand across jurisdictions, but the coverage of different jurisdictions is still uneven, and the proportion of samples related to the Civil Code is significantly higher.

### 18.4 Preference and risk data

* QA review record: `7731`
* High risk rejection sample: `6`
* Average QA rating: `5.0`

This shows that the project is not only building the main SFT data, but also building an auxiliary supervision layer related to QA, preferences and risk boundaries. For industry SFTs, this is often more important than simply increasing the total sample size.

### 18.5 Training and delivery layer products

* Training set split: `train = 6947`, `val = 790`, `smoke = 24`
* `training_manifest.json` Encapsulation completed
* The project check link passes, and the training and reporting side products can be aligned with each other.

This shows that the output of the project is no longer "several JSONL files", but a set of assets that can be directly consumed by the training side and verified by the check script consistency.

![图 16：P02 核心指标总览图](../../images/part10/10_2_fig16_metrics_dashboard.png)

---

## 19. Lightweight downstream verification: minimal verification design

In the previous version, P02 can already indicate that "the data factory is running smoothly." But it is not enough to prove that the process exists. It is more important to answer the following question:

> Is this data that has undergone QA, preference construction, and risk governance really better than unprocessed or low-quality candidates?

The new lightweight downstream verification is precisely to solve this problem.

### 19.1 Verify the design

The project randomly samples 50 paired samples under a fixed random seed `seed = 20260409`, and performs lightweight quality verification on two types of data: `chosen` and `rejected`. The goal here is not to pursue full model benchmarking, but to quickly answer whether preferences and QA actually drive the quality gap in a cost-controllable, reproducible way.

This verification method is suitable for the project goals at the current stage. The focus here is on data engineering methods rather than a complete downstream model paper. A good downstream validation does not have to be heavy to begin with, but it should meet at least three requirements:

* can be reproduced;
* able to explain;
* Can directly correspond to the previous data design assumptions.

### 19.2 Verification indicators

In this 50-item sampling, the project used several very representative categories of indicators:

* `chosen` Average quality score
* `rejected` Average quality score
* Pair winning rate
* Law citation coverage
* Unsafe shortcut expression rate

The nice thing about these metrics is that they are not purely abstract scores, but directly related to the real goals of legal SFT:

* Quality scores reflect overall acceptability;
* Pair winning rates reflect whether the preference construct actually differentiates between good and bad answers;
* The legal article citation coverage reflects whether the answer retains the legal basis;
* The expression rate of unsafe shortcuts reflects whether high-risk and arbitrary expressions are effectively suppressed.

### 19.3 Verification results

Judging from the current results:

* `chosen` Average quality score: `5.0 / 5`
* `rejected` Average quality score: `1.0 / 5`
* Match winning rate: `100.00%`
* Legal article citation coverage: `chosen = 100.00%`, `rejected = 0.00%`
* Unsafe shortcut expression rate: `chosen = 0.00%`, `rejected = 100.00%`

Although this set of numbers comes from light sampling, it illustrates very clearly:

First, the current preference structure and QA mechanism are not formally "reviewed", but actually distinguish high-quality samples from low-quality samples.
Second, two goals that are critical in legal scenarios—preserving legal basis and avoiding arbitrary shortcuts—show significant differences between `chosen/rejected`.
Third, the supervised design of P02 began to possess a very important engineering feature: not only can it produce samples, but it can also initially prove why these samples are worthy of being included in the training set.

### 19.4 The difference between lightweight verification and heavy benchmarking

It should be emphasized that the downstream verification here is still **lightweight verification**, not a complete training benchmark, nor is it a paper-oriented ablation research. Its value does not lie in replacing large-scale reviews, but in filling in the weakest piece of the puzzle for this chapter in the past:

> From "the data structure is reasonable" to "the data effect is supported by preliminary evidence".

This step is already very important. Because the biggest problem with many data engineering projects is that they only talk about the construction process without any post-test verification. In the end, it is difficult to judge whether the output data is really valid.

### 19.5 The engineering implications of this set of results

This lightweight verification brought at least three engineering-level signals.

First, show that preference pairs and QA records are worth keeping, not dispensable appendages. They not only help screen samples, but also provide clearer behavioral boundaries for subsequent training.

Second, it shows that the design of “explicit legal basis” is effective. The `chosen` sample reaches 100% legal reference coverage, while the `rejected` is 0%, indicating that the current template and quality inspection mechanism can significantly promote the model output to be more like real legal explanations, rather than general talk.

Third, it shows that "security boundary explicitness" has begun to take effect. The expression rate of unsafe shortcuts is 0% for `chosen` and 100% for `rejected`, which means that the project has been able to actively suppress high-risk expressions at the data level instead of leaving this matter entirely to the system prompt during inference.

### 19.6 How to interpret such experiments

The point of this type of experiment is not to claim a certain extreme result, but to illustrate three things:

* A minimum reproducible downstream verification has been added;
* It directly validates the key design assumptions presented earlier;
* It provides direction for subsequent heavier training experiments, rather than trying to solve all evaluation problems at once.

![图 17：50 条抽样验证流程图](../../images/part10/10_2_fig17_eval_sampling_protocol.png)

---

## 20. Interpretation of results: Structural signals of current data factories

Merely listing the results has limited meaning. What's more important is understanding the state of engineering reflected in these numbers.

### 20.1 From 2577 to 7737, indicating that the factory has the ability to expand

This shows that the project has achieved expansion from knowledge seeds to multi-task supervision data, rather than staying at the step of "organizing regulatory texts".

### 20.2 The three types of tasks are balanced, indicating that the task framework is stable

If there are far more tasks of a certain type than others, it usually means there is an imbalance in the template system or sampling logic. The current number of three types of tasks is completely aligned, indicating that the task distribution layer has high controllability.

### 20.3 The uneven distribution of jurisdictions shows that the focus of the next stage is not to continue to expand, but to supplement coverage.

The main conflict at present is no longer “whether there are samples”, but “whether the sample coverage is balanced”. This is more important than simply continuing the heap count.

### 20.4 Preferences, QA and lightweight downstream verification show that the project is moving from "answering well" to "answering more stably"

If there is only subject SFT, it is difficult to prove whether the model learned a better legal behavior pattern; if there are only preference pairs without verification, it is also difficult to prove whether the preference construction really plays a role. The newly added 50 lightweight downstream verifications provide a direct post-verification evidence for preference and QA: `chosen` and `rejected` have been significantly separated in terms of quality, citation and security.

In other words, the current project is beginning to have a more mature data engineering characteristics:

> It can not only generate training samples, but also prove "what kind of samples are more worthy of training" through minimal verification.

### 20.5 The cost of human review is already visible, indicating that automation and collaborative optimization must be considered in the future

The human review cost of 193.28 hours is not low in a small-scale demonstration project. Otherwise, it’s easy to mistakenly think that “generating a little more legal data just costs a little more API money.” In fact, what is really expensive is often the subsequent review and rework.

### 20.6 What changes does this supplement bring?

The newly added lightweight downstream verification completes the link of "how to judge whether the data design is effective".

Therefore, this chapter is no longer just "Legal SFT Data Factory Process Description", but forms a more complete engineering closed loop:

* Have goals and boundaries;
* There are data links and mission designs;
* There is QA, preference and risk control;
* There are training interfaces and inspection closed loops;
* There is also minimum reproducible downstream verification to support previous design judgments.

---

## 21. Quality Baseline: Usable Standards for Legal SFT Data

The quality baseline here is not to pursue an abstract full score standard, but to clearly state: what kind of data is enough to enter training, and what kind of data must continue to be reworked.

This type of project needs to establish at least the following four baselines.

### 21.1 Correctness Baseline

The answer cannot obviously violate the meaning of seed regulations, cannot add key conclusions out of thin air, and cannot omit applicable conditions to the extent that it affects the conclusion.

### 21.2 Expressing the baseline

Answers should be clear, complete, and minimally ambiguous. Even if it is legal terminology, it should be as readable as possible for non-professional users rather than mechanically copying legal provisions.

### 21.3 Format Baseline

Similar tasks should follow a consistent output skeleton. For example, the case analysis category should try to include issues, rules, analysis and conclusions, rather than sometimes being a paragraph or sometimes a set of fragments.

### 21.4 Risk baseline

High-stakes issues must reflect a sense of boundaries. When encountering questions with insufficient evidence, incomplete facts, or questions that are obviously close to the opinion of the individual case, the answer should be carefully expressed or trigger a rejection template, rather than forcing a clear judgment.

### 21.5 Baseline and single scores

Compared with abstractly saying "the overall quality is good", the quality baseline is more like a threshold: if you pass the line, you can move to the next step; if you don't, you must rework. This has more engineering value than a single average score.

---

## 22. Version evolution: version management of industry SFT data sets

A mature data factory will not regard the first version of the data set as the final answer. Instead, it should naturally support version evolution.

### 22.1 V1: Clean and cut the regulations first

The value of the first version is to establish a stable link from PDF to structured seed. It answers: Can the seed layer be reliably generated?

### 22.2 V2: Introducing three types of main tasks

The value of the second version is to move data from "knowledge fragments" into "supervision samples". This version solves the problem of task system and distributed control.

### 22.3 V3: Add preference pairs and review records

The value of the third version is that the sample can not only train "correct answers", but also train "better answers", and make the quality history traceable.

### 22.4 V4: Add risk rejection and online boundaries

The value of the fourth edition is to separately model high-risk behaviors so that factories have the most basic compliance and safety awareness capabilities.

### 22.5 Record value of version evolution

It can explain very intuitively: the data factory does not grow all at once; each version has its own core goals; not all problems should be solved in the first version; the triggering conditions for version upgrades should come from real problems rather than abstract perfectionism.

![图 18：P02 版本演进路线图](../../images/part10/10_2_fig18_version_timeline.png)

---

## 23. Cost Optimization: Key Cost Items of Legal Data

When many teams are working on large model data projects, the first thing they think of is the cost of the model API. But in legal SFT, the really expensive part is usually not the generation, but:

* Manual review
* Error rework
* High-risk sample upgrade processing
* Version regression check

### 23.1 Cost Implications for Current Projects

The human review time of the current project has reached 193.28 hours. This number itself is a very good reminder: the industry data factory is never a question of "run the model more", but a question of "how to prevent human-machine collaboration from getting out of control."

### 23.2 Which links deserve the most priority for automation?

In legal scenarios, the priority for automation is usually not the final arbitration, but the previous mechanical screening, for example:

* Samples with unqualified formats will be eliminated;
* Pre-qualification of low-risk template samples;
* Rule interception of expressions that are obviously out of bounds;
* Clustering of error types for review records.

### 23.3 The need for cost analysis

Because the project part not only shows "the method is feasible", it should also explain the input-output relationship. If a solution is good in theory, but cannot bear the human cost in reality, then it is not a truly implementable method.

---

## 24. Validation closed loop: consistency check of legal data pipeline

Whether a project is mature or not depends not only on whether there are output files, but also whether there is consistency verification.

### 24.1 Check what the script does

Because industry data projects are prone to the problem of "partial parts are correct, but the whole does not make sense". for example:

* The code can run, but the product is missing files;
* The number of samples looks normal, but there is a leak in train/val;
* The metrics says passed, but the report quotes old numbers;
* The number of preference pairs does not match the number of subject samples;
* smoke test does not represent the real training format.

### 24.2 Current project verification status

The project inspection results are:

* Total inspection items: 13
* Passed check items: 13
* Overall status: PASS

Command-level inspections cover `py_compile`, `evaluate_factory`, etc.; data/product-level inspections cover key projects such as `required_files_exist`, `seed_count_positive`, `accepted_count_matches_seed_x_tasks`, `preference_pairs_cover_accepted`, `qa_reviews_cover_accepted`, `train_val_no_overlap`.

### 24.3 Verify the engineering role of closed loops

Because it embodies a very important engineering habit: the completion standard for data projects is not "it seems that a lot of files are generated", but "the code, products, statistics and reports are consistent with each other."

![图 19：代码—产物—报告一致性验证图](../../images/part10/10_2_fig19_validation_chain.png)

---

## 25. Limitations and Risks: Current Factory Constraints

A case that only talks about success is usually not credible enough. This is especially true for legal SFT, as it is naturally subject to data sources, audit costs, and risk boundaries.

### 25.1 Uneven jurisdictional coverage

The current sample is significantly biased towards a small number of legal jurisdictions, which will lead to uneven performance of the model in terms of knowledge breadth. One of the most important tasks in the next stage is to complete the high-frequency issues of long-tail jurisdictions and real business.

### 25.2 Higher synthesis ratio

Although synthesis is a necessary means for cost control, too high a synthesis ratio will cause template cavity and teacher offset problems. The model may learn to "answer like a template", but it may not truly grasp the diverse user expressions.

### 25.3 The number of risk refusal samples is still too small

A risk rejection mechanism has been established, but the sample size is still small. For real online scenarios, this is far from enough. Especially in scenarios such as individualized suggestions, sensitive disputes, and insufficient evidence judgment, more abundant rejection and retention boundary samples are needed.

### 25.4 QA is expensive

As the sample size expands, the cost of human review will continue to rise. If a more fine-grained pre-review, arbitration and re-bid mechanism is not introduced into the process, subsequent expansion will face significant resistance.

---

## 26. Cross-industry migration: the model value of legal factories

Law isn't the only industry that needs vertical SFT, but it's a great example. The reason is that the legal scene has the following characteristics at the same time:

* Strongly structured knowledge
* Strong task constraints
* Clear risk boundaries
* QA demand rigidity
* High cost of human-machine collaboration

These characteristics actually also exist in industries such as taxation, finance, medical care, and customer service and compliance.

### 26.1 Directly transferable design

* Cleaning link from unstructured documents to structured seeds;
* The practice of splitting the task system first and then expanding it;
* The idea of ​​​​parallel construction of SFT, preference pairs, and risk rejection;
* QA protocols, error labeling and rework mechanisms;
* Training encapsulation and verification closed loop.

### 26.2 Parts that cannot be copied directly

* Risk boundaries in legal scenarios are not equal to medical or financial boundaries;
* Legal interpretation tasks may not be the main task in other industries;
* Legal document style does not equal customer service or sales style;
* Trigger conditions for high-risk rejections need to be rewritten by industry.

### 26.3 Migrating method chains

What can really be migrated is not a specific prompt, but this method chain:

> Find authoritative seeds -> do structured dicing -> design task system -> controlled synthetic expansion -> establish QA and preferences -> individually model risk boundaries -> train encapsulation and consistency verification.

![图 20：行业迁移方法链图](../../images/part10/10_2_fig20_cross_domain_transfer.png)

---

## 27. List of major deliverables

A list of the main deliverables is given here.

### 27.1 Seeds and processing intermediates

* `data/processed/raw_chunks.jsonl`
* `data/processed/legal_seed_dataset.jsonl`
* `data/processed/instruction_taxonomy.json`

### 27.2 Main supervision and auxiliary supervision products

* `data/processed/domain_expert_sft.jsonl`
* `data/processed/synthetic_candidates_rejected.jsonl`
* `data/processed/legal_preference_pairs.jsonl`
* `data/processed/legal_qa_review.jsonl`
* `data/processed/legal_risk_refusal_sft.jsonl`
* `data/processed/legal_risk_register.jsonl`

### 27.3 Training interface products

* `data/training/final_sft_dataset.jsonl`
* `data/training/train.jsonl`
* `data/training/val.jsonl`
* `data/training/smoke_test.jsonl`
* `data/training/training_manifest.json`

### 27.4 Reporting and Verification Products

* `data/reports/p2_report.md`
* `data/reports/p2_metrics.json`
* `data/reports/p2_test_results.json`

Listing these deliverables is not just to present a list, but to illustrate: the result of an industry SFT factory is not just a training set file, but a set of interrelated data assets.

---

## 28. Summary: Organize generation into factories

Looking back at the entire project, you will find that the problem it really solves is not "how to make the model generate more legal questions and answers", but:

* How to extract usable knowledge from authoritative but dirty PDFs;
* how to break down knowledge into different types of supervisory tasks;
* How to make the generation process controllable instead of random expansion;
* How to make model behavior boundaries into data through preference pairs and risk rejections;
* How to incorporate QA, cost, version and verification closed loops into factory design.

This is also the thing I want to convey most in this chapter:

> In highly professional industries, the goal of SFT data engineering has never been to “make more samples”, but to establish an assembly line that can continuously and stably produce high-quality surveillance assets.

Law is just one representative scenario of this approach. As long as the team masters this link from seed to supervision, from generation to quality inspection, and from sample to online interface, the team will have the basic method base for building data factories in other industries.

---

## Special Topic: Legal Release Gating of SFT Data

One of the biggest differences between the legal field data factory and the general question and answer data factory is that it naturally bears a higher error cost. For general Q&A, insufficient answers may simply be a poor experience; for legal Q&A, wrong supervision signals are likely to be directly amplified to risk recommendations, rejection boundaries, and professional credibility. Therefore, projects such as P02 need clear access control conditions when entering version release.

### 1. Before publishing, you need to look at both content risks and engineering risks.

Legal SFT data cannot only look at "whether the data volume is sufficient" or "the format is uneven", but also needs to be judged at the same time:

* Whether there are obvious errors or outdated expressions in the citations of legal provisions, case summaries and risk warnings;
* Whether rejection samples and high-risk consultation samples cover key boundaries;
* Whether the preference pairs and QA records actually support the current version of the conclusions;
* Whether the training interface, manifest and test results are consistent with the version product.

In other words, the release threshold of the legal data version is naturally higher than that of ordinary data, because it must pass both the content credibility check and the engineering consistency check.

### 2. The value of access control lies in making “caution” a system attribute.

When many teams produce industry data, they leave caution to manual review at a later stage, but a safer approach is to shift caution to release gates. As long as the access control is written down in a structured way, the team will gradually form a stable habit: instead of publishing first and then adding explanations, they first confirm the risk boundaries, monitor assets and verify evidence, and then decide whether to enter training and display for this version. This kind of institutionalized caution is precisely the long-term capability that is most worth retaining in industry SFT data.

