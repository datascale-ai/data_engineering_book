# Chapter 38: StructBill-CN Bill Document Understanding Data Engineering

## Chapter Summary

In high-risk document scenarios such as medical bills, settlement statements, and pharmacy invoices, “reading the text in the image” is far from enough. The real target is to transcribe the image directly into a queryable and ingestible database record. Current multimodal large models face three intertwined difficulties in this task: global key-value extraction often produces numeric hallucination and spatial drift; traditional table structure recognition fails on borderless tables; and token-level losses cannot perceive arithmetic inconsistency where two strings differ by only one character but one is completely wrong for the business logic. A syntactically valid JSON object with mostly correct fields may still be arithmetically self-contradictory and therefore unusable for financial audit or insurance claim systems.

This chapter uses the StructBill-CN dataset to explain how to build a trainable, evaluable, and reviewable data asset from real bill images, predefined schemas, hierarchical JSON annotations, and deterministic logic constraints. The focus is not the dataset name itself, but four engineering questions: why high-risk documents are difficult to extract structurally; how sample schema encodes schema, hierarchical JSON, and logic constraints in one annotation; how the construction pipeline uses consistency gates to keep labels arithmetically self-consistent; and how the evaluation protocol closes the loop across field accuracy, structural consistency, arithmetic consistency, and schema-constraint violation.

As the first case in Part 12, this chapter connects document understanding and OCR from Part 3, RAG and document evidence organization from Part 7, data versioning and experiment tracking from Part 8, data assets and contract governance from Part 9, and compliance boundaries from Part 11. It also prepares a Chinese vertical-document engineering example for the VLM data recipes in Part 13 and the multimodal RAG / privacy-pipeline projects in Part 14.

## 38.0 Learning Objectives

After studying this chapter, you should be able to:

- Explain why Chinese bills and medical-expense documents are a high-risk, high-density, weak-visual-cue data engineering challenge rather than merely an OCR problem.
- Understand StructBill-CN's task definition: Schema-based End-to-End Unified Extraction, and how it differs from traditional table structure recognition.
- Understand the three supervision signals in each sample: global key-value fields, nested line-item tables, and logic constraints.
- Apply a construction pipeline with logic-consistency gates across acquisition, denoising, schema design, hierarchical JSON annotation, schema validation, logic validation, and version splitting.
- Use multidimensional metrics such as KV-F1, Table-F1, ANLS, TEDS, ACR, and SCVR as a coherent evaluation loop.
- Attribute errors to concrete data-engineering repair actions.
- Recognize privacy, compliance, and audit requirements in high-risk document scenarios, and understand the principle of public benchmark / private production separation.
- Connect this chapter with later VLM data recipes, multimodal RAG, and privacy-pipeline projects.

## Scenario Introduction

A provincial medical-insurance center spent two months training an end-to-end bill extraction system based on Qwen2.5-VL (Bai et al., 2025). Offline evaluation looked promising: character-level recognition accuracy measured by ANLS exceeded 92%, and field-level F1 approached 90%. The team was ready to connect the system to the settlement reconciliation pipeline.

Before launch, however, the finance-accounting group randomly sampled 200 borderless expense lists for cross-checking. Nearly 15% of the records failed the rule “sum of line-item amounts = total amount.” Most failures were not ordinary character-recognition errors. The numbers themselves were read correctly, but row and column drift assigned amounts to the wrong line. Even worse, about 5% of lists contained fabricated table rows: the model turned free-text discharge-record paragraphs into plausible but nonexistent fee items.

The acceptance team asked three questions. First, did the existing test set check “unit price x quantity = amount” row by row? If labels themselves are not checked for arithmetic consistency, how can a model learn the rule? Second, if evaluation only reports ANLS and F1, is there a metric for “how many records can be inserted into the database without manual review”? Third, if an image is so degraded that it is unreadable, is that a model error or a data error?

These questions hit the core issue. A dataset that only labels where fields are located cannot expose arithmetic inconsistency, structural fabrication, or row-column drift. An evaluation protocol that only measures character-level accuracy cannot answer whether a record is usable. The team must return to data engineering: annotation rules, validation pipelines, and evaluation protocols. StructBill-CN is designed around exactly this problem.

## 38.1 Why Bills and Medical-Expense Documents Are Hard to Extract

Medical bills, settlement statements, and pharmacy invoices sit at the intersection of high risk, high density, and weak visual cues. They are not free text that can be paraphrased, nor clean electronic spreadsheets with visible grids. They are structured records that must enter downstream systems such as financial audit, insurance claim, and ERP systems. The goal is therefore not to read text from an image, but to transcribe the image into a queryable database object.

From a data-engineering view, the difficulty has three layers.

The first layer is unreliable **global key-value extraction**. When a model retrieves fields such as total amount or invoice number, it often hallucinates digits or drifts spatially to a neighboring row or column (Liu et al., 2024). A small numeric drift may be harmless in free-text description, but in financial records it creates unusable dirty data.

The second layer is that **traditional table processing fails**. Much table structure recognition assumes visible grid lines and physical coordinates. Real bills often contain borderless tables: no vertical separators, dense numeric columns, and visually sticky columns. Segmentation-based methods drift easily. More importantly, TSR outputs physical structure, while business systems need semantic schema; a nontrivial gap remains.

The third and most easily missed layer is **logic consistency**. Bills contain deterministic arithmetic axioms: unit price x quantity = amount, and sum of line-item amounts = total amount. These constraints are nearly invisible to token-level loss. An incorrect amount may differ from the correct one by one character, but business logic treats it as wholly wrong. A JSON output can be syntactically legal and mostly field-correct while still contradicting itself arithmetically.

This is why the problem is data engineering, not only modeling. To make a model learn semantic layout and arithmetic constraints, the dataset must explicitly encode image, schema, hierarchical JSON, table fields, and logic constraints in a trainable, evaluable, and reviewable asset.

### 38.1.1 Pipeline vs. End-to-End Extraction

Two technical routes dominate document extraction. The **pipeline paradigm** separates text detection, OCR recognition, and information extraction. It is modular and interpretable, but its fatal weakness is error accumulation. A wrong detection box propagates irreversibly to downstream extraction.

The **end-to-end generative paradigm** asks one multimodal model to generate structured output directly from the image. This avoids some cascade errors, but introduces a new problem: general models tend to produce fluent descriptions rather than strict database records, causing formatting errors and missing key information.

StructBill-CN takes the end-to-end side, but uses schema constraints to force fluent description back into strict records and logic constraints to force plausible outputs into arithmetic correctness. The dataset itself must carry both constraints; otherwise the model has no signal to learn. This is its fundamental difference from datasets such as FUNSD and DocVQA that mainly label key-value pairs or physical boxes.

### 38.1.2 How One Drift Cascades into an Invalid Record

Bill errors are not isolated. Imagine a borderless expense list with sparse empty values. Without grid lines, the model shifts one row and assigns the amount in row 3 to row 2. Every digit may be recognized correctly, and ANLS may remain high. But row-level “unit price x quantity = amount” fails, and document-level “sum of items = total” also fails. A record that looks almost perfect at the character level becomes unusable for the business.

This is why arithmetic self-consistency must be a first-class object. In high-risk structured extraction, the unit of data quality is not the character; it is the database-ingestible record.

## 38.2 Dataset Overview: Scale, Sources, Schema, and Task Definition

### 38.2.1 Scale and Sources

StructBill-CN contains **2,300 high-resolution bill images** across **six business schemas**, all from two public medical datasets: CHIP-2022 and SIBR-Med. The mixture intentionally includes wired-grid tables, text-dense records, and borderless tables so the model cannot simply memorize one layout.

*Table 38-1: StructBill-CN composition and characteristics*

| Source Subset | Document Type | Count | Table Form |
| --- | --- | ---: | --- |
| CHIP-2022 | Inpatient invoice | 680 | Wired grid |
| CHIP-2022 | Outpatient invoice | 340 | Wired grid |
| CHIP-2022 | Pharmacy invoice | 340 | Wired grid |
| CHIP-2022 | Discharge record | 340 | Text-dense |
| SIBR-Med | Expense list | 400 | Borderless table |
| SIBR-Med | Notice form | 200 | No table |
| **Total** | **6 schemas** | **2,300** | **Mixed** |

Using public academic sources is a deliberate compliance choice. A publishable benchmark should be built on public sources, while real private production data should enter only through a governed production process. This public benchmark / private production split is a baseline principle for high-risk document data engineering.

### 38.2.2 Task Definition

Given a document image $X$ and a schema $S=\{K,T,C\}$, where $K$ is the set of global key fields, $T$ is the table definition, and $C$ is the set of deterministic constraints, the goal is to learn a policy that generates a structured sequence $Y$ maximizing $P(Y\,|\,X,S)$.

Unlike ordinary end-to-end text generation, this task requires output that strictly follows predefined structure and business logic. It must be correct in content, legal in structure, and self-consistent in arithmetic. The three parts of $S$ turn an OCR/extraction task into a structure-and-logic constrained extraction task.

### 38.2.3 Three Core Challenges

StructBill-CN deliberately keeps three types of difficulty.

**First, explicit visual cues are missing.** Borderless tables contain no vertical separators, causing dense numeric columns to stick together visually. Annotation must cut columns semantically rather than geometrically, and quality review must inspect column ownership.

**Second, structural ambiguity and hallucination are common.** Free-text blocks can induce fabricated table rows, while sparse empty columns can shift entire rows. The schema must declare anti-hallucination constraints, and the annotation rules must define empty placeholders and alignment rules.

**Third, density and visual noise are extreme.** Real documents produce long sequences, physical degradation, and semantically similar fields. Acquisition should bucket image quality early so “the image is unreadable” can be separated from “the model read it incorrectly.”

These difficulties are design targets, not defects. They explain why the construction pipeline needs logic gates, quality grading, and semantics-first annotation.

## 38.3 Sample Schema: Key-Value, Line-Item Tables, Hierarchical JSON, and Logic Constraints

### 38.3.1 Three Supervision Signals

Each StructBill-CN sample pairs one bill image with a predefined schema and carries three complementary supervision signals:

1. **Global key-value structure:** document-level attributes such as hospital name, invoice number, and total amount.
2. **Nested line-item table:** row-level fields such as item name, unit price, quantity, and amount.
3. **Schema-bound logic constraints:** deterministic arithmetic rules for numeric fields.

The key annotation philosophy is **semantic ownership over physical position**. When layout drift or borderless tables make geometry misleading, labels are assigned by business context rather than pixel location. This makes annotation harder and demands domain understanding, but forces models to learn content logic rather than shallow projection.

Hierarchical JSON is used because it maps directly to real database schema: global attributes plus nested line items. Flat key-value pairs cannot express one-to-many detail rows, while physical coordinates leave semantics for downstream systems to infer. Hierarchical JSON is the natural form for the ingestion-ready goal.

### 38.3.2 Mapping Schema to Hierarchical JSON

The three schema parts map to the final JSON as follows: $K$ becomes the global `key_information` object, $T$ becomes the `Fee_List` array and its row fields, and $C$ becomes validation relationships attached to numeric fields rather than visible JSON nodes.

```mermaid
flowchart LR
  S["Schema S = tuple K, T, C"] --> K["K: global key fields"]
  S --> T["T: table definition"]
  S --> C0["C: deterministic constraints"]
  K --> J1["key_information<br>Hospital_Name<br>Invoice_No<br>Total_Cost"]
  T --> J2["Fee_List array<br>Item_Name / Unit_Price<br>Quantity / Amount"]
  C0 --> J3["logic binding<br>unit price x quantity = amount<br>sum(amount) = total"]
  J1 --> O[("hierarchical JSON output")]
  J2 --> O
  J3 -. validate .-> O
```

*Figure 38-1: Schema-to-JSON mapping. Key fields and table structure become visible JSON nodes; constraints remain verifiable relationships attached to numeric fields.*

This “constraints as relationships, not fields” design lets the same JSON serve training and evaluation. Constraints do not change the output format, but they are instantiated during validation as equations. A future schema can add a new rule such as discounted amount = amount x discount rate without changing historical fields.

### 38.3.3 Complete Sample Structure

```json
{
  "key_information": {
    "Hospital_Name": "<hospital_name>",
    "Invoice_No": "4700852972",
    "Total_Cost": 699.02
  },
  "Fee_List": [
    {
      "Item_Name": "<item_a>",
      "Unit_Price": 54.76,
      "Quantity": 1.00,
      "Amount": 54.76
    },
    {
      "Item_Name": "<item_b>",
      "Unit_Price": 2.10,
      "Quantity": 2.00,
      "Amount": 4.20
    }
  ]
}
```

In this small sample, `key_information` and `Fee_List` are structure. The row-level and document-level arithmetic equations are logic constraints. Both must be annotated, validated, and evaluated.

### 38.3.4 Field Types, Annotation Rules, and Metrics

*Table 38-2: Field type, annotation rule, and metric mapping*

| Field Type | Representative Fields | Annotation Rule | Main Metric |
| --- | --- | --- | --- |
| Text attribute | `Hospital_Name`, `Item_Name` | Semantic ownership first; tolerate minor OCR noise in long text | ANLS / Entity-Level F1 |
| ID / string | `Invoice_No` | Exact transcription; preserve leading zeros and separators | Exact-match F1 |
| Numeric attribute | `Unit_Price`, `Quantity`, `Amount` | Standardized decimal format; bind row-level arithmetic | Exact-match F1 + Row-ACR |
| Global total | `Total_Cost` | Bind to line-item sum | Doc-ACR |
| Structure / topology | `Fee_List` row set | Row alignment; keep empty values with placeholders | TEDS / Table-F1 |

This table acts as a contract between annotation rules and evaluation scripts.

## 38.4 Construction Pipeline

StructBill-CN uses a multi-stage pipeline whose goal is to preserve semantic content and business-logic topology while creating traceable quality gates.

```mermaid
flowchart TD
  A["1. Collect raw bill images<br>CHIP-2022 / SIBR-Med"] --> B["2. Denoise and grade image quality<br>dedup / deskew / resolution filter"]
  B --> C1["3. Design schema<br>global keys K + table T + constraints C"]
  C1 --> D["4. Annotate hierarchical JSON<br>global KV + nested line items"]
  D --> E["5. Validate schema alignment<br>required keys / types / structure"]
  E --> F["6. Validate logic consistency<br>row product / document sum"]
  F --> G{"quality gate passed?"}
  G -->|no| D
  G -->|yes| H["7. Version split<br>Train : Test = 8 : 2"]
  H --> I[("trainable, evaluable,<br>reviewable data asset")]
```

*Figure 38-2: StructBill-CN construction pipeline. Samples that fail logic validation return to annotation rather than entering the training set.*

**1. Acquisition.** Images come from CHIP-2022 and SIBR-Med and intentionally include borderless tables, sparse layouts, and long expense lists.

**2. Denoising and quality grading.** This stage should include duplicate removal, skew/rotation correction, filtering or bucketing of low-resolution and severely damaged images, and image-quality metadata for later error attribution.

**3. Schema design.** For each business document type, define $S=\{K,T,C\}$. The schema is the data contract: annotation rules and evaluation scripts follow it once it is frozen.

**4. Hierarchical JSON annotation.** Ground truth is organized as global key-value attributes plus nested line items. Labels are assigned by semantic ownership rather than geometry. For no-table documents, `Fee_List` can be empty while the schema remains consistent.

**5. Schema alignment validation.** The first automatic gate checks that JSON is parseable, required root and table keys exist, and field types match the schema.

**6. Logic consistency validation.** The core step checks whether annotations are arithmetically self-consistent: row by row, unit price x quantity approximately equals amount; at document level, line-item amounts approximately sum to total. Tolerance $\varepsilon$ absorbs OCR and floating-point noise.

```mermaid
flowchart TD
  P["input: annotation or prediction JSON"] --> G1{"JSON parseable?"}
  G1 -->|no| X["structure gate failed<br>I_gate = 0, reward = 0"]
  G1 -->|yes| G2{"schema-compliant?"}
  G2 -->|no| X
  G2 -->|yes| G3{"fabricated table rows?"}
  G3 -->|yes| X
  G3 -->|no| R1["row-level check<br>each u * q ~= a ?"]
  R1 --> R2["document-level check<br>sum(a) ~= T ?"]
  R2 --> SC["consistency score"]
```

*Figure 38-3: Logic-consistency validation. The same gate is reused during construction to block inconsistent labels and during evaluation/training to score model output.*

**7. Version split.** The dataset uses an 8:2 train-test split. In practice, the split should preserve the six schema distributions, reserve true cross-layout test samples, and attach data fingerprints and statistics to each version.

### 38.4.1 Lineage and Metadata

Producing only images and JSON is not enough. Each sample should carry lineage metadata: source subset and original file ID, schema version, image-quality grade, annotator and reviewer, pass/fail results and tolerances for each logic check, and final split.

This metadata supports error attribution, audit compliance, and reproducibility. It is also crucial for SFT warm-start data: if the supervised data used before reinforcement learning contains illegal JSON structures, the RL stage loses a stable starting point. The schema gate must therefore be strict for warm-start data.

## 38.5 Evaluation Protocol

StructBill-CN evaluates **extraction accuracy**, **structural quality**, and **logic consistency**.

- **KV-F1 / Table-F1 (Entity-Level F1):** precision and recall for global key fields and table fields.
- **ANLS:** character-level accuracy for long text with tolerance for small OCR noise.
- **TEDS:** tree-edit-distance similarity for generated JSON topology, especially nested tables.
- **ACR:** Arithmetic Consistency Rate, composed of **Row-ACR** for row-level unit-price checks and **Doc-ACR** for document-level sums.
- **CHIP-2022 Score:** official macro-F1 on the public CHIP-2022 subset, with exact-match F1 for categorical/numeric fields and normalized edit distance for text-dense fields.

These metrics must coexist. F1 says whether fields were found, but not whether numbers add up. ANLS tolerates long-text noise but does not guarantee arithmetic. TEDS captures structural collapse but not numeric correctness. Row-ACR and Doc-ACR directly answer whether the numbers reconcile.

### 38.5.1 Schema Constraint Violation Rate

Academic metrics tend to be positive: how much is correct. Production monitoring often needs the negative view: how much violates constraints. From the gate in Section 38.4, we can derive **SCVR (Schema Constraint Violation Rate)**: the proportion of outputs that fail the structure gate or logic validation. SCVR complements Row-ACR and Doc-ACR by answering how many records cannot be inserted directly, including structural failures.

SCVR adds no new labels. It reuses the existing structure and logic validation flow.

### 38.5.2 Engineering Conventions for Reproducible Evaluation

Reproducible evaluation requires a fixed test split fingerprint, fixed schema version, fixed metric implementation, and controlled random seeds. For generative models, decoding parameters and repeated runs matter. The source setting uses `temperature=0.9`, `top_p=1.0`, and reports the average over **8 independent runs** per model to absorb decoding variance. Engineering teams should archive decoding parameters, run count, and seeds with the results.

### 38.5.3 Error Attribution and Repair Actions

*Table 38-3: Common errors and repair actions*

| Error Type | Symptom | Root Cause | Data-Engineering Repair |
| --- | --- | --- | --- |
| Numeric hallucination | Amount or quantity is fabricated or copied wrongly | Token-level approximation, missing logic constraints | Bind P x Q = A and sum = T; use Doc-ACR as a quality gate; create numeric negatives |
| Spatial drift | Field value comes from neighboring row or column | Borderless table without grid lines | Semantic-ownership annotation; column-stable anchor review; record column ownership |
| Fabricated row | Free text becomes table row | Unstructured text induces hallucination | Hallucination gate (`I_gate=0`); mark schema with `anti_hallucination` |
| Row drift | Sparse empty values shift the row | Missing empty placeholder | Annotate empty placeholders; Hungarian row matching review; bucket empty-column samples |
| Illegal structure | JSON cannot parse or required key is missing | Free generation without constraints | Schema gate; pre-run structure validator; freeze schema version |
| Broken total | Line-item sum does not equal total | Missing document-level check | Document-level consistency validation; send over-tolerance samples back for relabeling |

The engineering value of attribution is speed. When a metric regresses, teams can decide whether the issue is image quality, annotation drift, or model capability. The table should also feed back into annotation rules: frequent online errors should trigger rule review and affected-sample relabeling.

## 38.6 Engineering Review

### 38.6.1 How the Data Asset Supports SRPO

This chapter does not detail the model. From the data-consumption perspective:

- **Data becomes reward.** SRPO converts the discrete schema rules in Section 38.3 into dense, verifiable SCL-Reward: $R_{total}=I_{gate}\cdot[\lambda\cdot R_{content}+(1-\lambda)\cdot R_{logic}]$. The structure gate, content alignment, and logic validation all read the dataset's hierarchical JSON and constraints.
- **Training use.** SRPO first uses the data for SFT warm start so the model can output legal JSON, then uses GRPO (Shao et al., 2024) with group sampling and SCL-Reward. The reported configuration is SFT for 10 epochs, learning rate 1e-5, batch size 128; GRPO group size G=8, reward coefficients $\lambda=0.4$ and $\gamma=0.6$; hardware 8 x NVIDIA A800 (80GB).
- **Qualitative effect.** The source material reports that standard SFT saturates near 84% on logic scores, while adding logic reward improves Row/Doc-ACR by about 10 percentage points. The point is that logic annotation turns arithmetic consistency into an optimizable target.

Hungarian matching (Kuhn, 1955) is used for row-level one-to-one alignment because generated row order may differ from ground truth or include missing/spurious rows. That, in turn, requires each row to contain sufficiently discriminative fields such as item name, unit price, and quantity. Algorithm design and annotation rules must be co-designed.

### 38.6.2 What the Dataset Is Suitable For

It is suitable for training **schema-constrained Chinese vertical-document extraction models**, especially small multimodal document models around the 3B scale that need robust alignment on borderless tables and sparse layouts. It is also suitable for evaluating logic consistency and structural fidelity, not merely character recognition.

### 38.6.3 Privacy, Compliance, and Audit in High-Risk Scenarios

Medical-expense documents are high-risk data. Even though this benchmark uses public academic sources, any production extension must follow these baselines:

**Privacy and de-identification.** Real bills and medical records must be de-identified and masked before entering the pipeline. Public benchmarks should use authorized, masked, or public sources.

**Human in the loop.** Extraction used for claims, audit, or database ingestion must retain human review. The model is an assistant, not an automatic decision-maker.

**Auditability.** Each record should trace back to source image version, schema version, annotator/reviewer, and logic-validation results. SCVR and the error-attribution table can form the audit chain of who changed what, in which version, and why.

### 38.6.4 Where Not to Use It

This dataset should not be used:

- to drive unattended clinical or automatic-claim decisions
- as a general OCR or layout-restoration benchmark
- in cross-language or cross-domain settings without redesigning schema and logic validation
- as the sole dataset mixed blindly with classification tasks, because arithmetic extraction and semantic classification can create negative transfer

### 38.6.5 Evolution from a Data Perspective

Two directions are important. First, **multi-task negative transfer**: mixing arithmetic extraction and semantic classification may create conflicting gradients and degrade numeric fields. Data mixtures, sampling temperature, and curriculum order must be tracked as hyperparameters. Second, **from deterministic constraints to adaptive reward**: current logic reward depends on hand-written arithmetic rules, but future systems may use data-driven adaptive rewards. The schema should therefore include a rule-version dimension so constraints can evolve while evaluation remains reproducible.

StructBill-CN should be treated as an evolvable data contract. Its schema, constraints, and splits are versioned and can grow with downstream VLM and RAG needs.

## Chapter Summary

StructBill-CN is not just another dataset. It addresses a data-engineering question: how to turn high-risk Chinese bill and medical-expense documents into a trainable, evaluable, and reviewable data asset from images, schemas, hierarchical JSON, table fields, and logic constraints.

The main conclusion is threefold. First, in high-risk structured extraction, data quality is measured by ingestible records, not characters. Character-level metrics alone cannot expose row-column drift, arithmetic inconsistency, and structural fabrication. Second, schema is the contract for annotation, the baseline for evaluation scripts, and the input to logic reward. Third, StructBill-CN is an evolvable data contract whose schema, constraints, and splits can keep growing with VLM data recipes, multimodal RAG, and privacy-pipeline projects.

## References

Bai, S., Chen, K., Liu, X., et al. (2025). Qwen2.5-VL Technical Report. *arXiv preprint arXiv:2502.13923*.

Blecher, L., Cucurull, G., Scialom, T., and Stojnic, R. (2023). Nougat: Neural Optical Understanding for Academic Documents. *arXiv preprint arXiv:2308.13418*.

Huang, Y., Lv, T., Cui, L., Lu, Y., and Wei, F. (2022). LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking. *Proc. ACM Multimedia*.

Huang, Z., Chen, K., He, J., Bai, X., Karatzas, D., Lu, S., and Jawahar, C.V. (2019). ICDAR2019 Competition on Scanned Receipt OCR and Information Extraction. *Proc. ICDAR*, pp. 1516-1520.

Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., and Chen, W. (2021). LoRA: Low-Rank Adaptation of Large Language Models. *arXiv preprint arXiv:2106.09685*.

Jaume, G., Ekenel, H.K., and Thiran, J.-P. (2019). FUNSD: A Dataset for Form Understanding in Noisy Scanned Documents. *ICDAR Workshop*.

Kuhn, H.W. (1955). The Hungarian Method for the Assignment Problem. *Naval Research Logistics Quarterly*, 2(1-2), pp. 83-97.

Levenshtein, V.I. (1965). Binary Codes Capable of Correcting Deletions, Insertions and Reversals. *Soviet Physics Doklady*, 10, pp. 707-710.

Liu, H., Xue, W., Chen, Y., et al. (2024). A Survey on Hallucination in Large Vision-Language Models. *arXiv preprint arXiv:2402.00253*.

Mathew, M., Karatzas, D., and Jawahar, C.V. (2021). DocVQA: A Dataset for VQA on Document Images. *Proc. WACV*.

Niu, J., Liu, Z., Gu, Z., et al. (2025). MinerU 2.5: A Decoupled Vision-Language Model for Efficient High-Resolution Document Parsing. *arXiv preprint*.

Park, S., Shin, S., Lee, B., et al. (2019). CORD: A Consolidated Receipt Dataset for Post-OCR Parsing. *NeurIPS Workshop on Document Intelligence*.

Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C.D., and Finn, C. (2024). Direct Preference Optimization: Your Language Model Is Secretly a Reward Model. *Proc. NeurIPS*.

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. (2017). Proximal Policy Optimization Algorithms. *arXiv preprint arXiv:1707.06347*.

Shao, Z., Wang, P., et al. (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. *arXiv preprint arXiv:2402.03300*.

Tianchi, A. and CHIP Committee (2022). CHIP 2022 Shared Task: Medical Invoice OCR Element Extraction Dataset. *Aliyun Tianchi Platform*.

Xu, Y., Li, M., Cui, L., Huang, S., Wei, F., and Zhou, M. (2020). LayoutLM: Pre-training of Text and Layout for Document Image Understanding. *Proc. ACM SIGKDD*, pp. 1192-1200.

Xue, W., Yu, B., Wang, W., Tao, D., and Li, Q. (2021). TGRNet: A Table Graph Reconstruction Network for Table Structure Recognition. *arXiv preprint arXiv:2106.10598*.

Yang, Z., Long, R., Wang, P., et al. (2023). Modeling Entities as Semantic Points for Visual Information Extraction in the Wild. *Proc. CVPR*.

Zhang, N., Chen, M., Bi, Z., et al. (2022). CBLUE: A Chinese Biomedical Language Understanding Evaluation Benchmark. *Proc. ACL*, pp. 7888-7915.

Zhong, X., ShafieiBavani, E., and Jimeno Yepes, A. (2020). Image-based Table Recognition: Data, Model, and Evaluation. *arXiv preprint arXiv:2011.13534*.

SRPO code: https://github.com/Yuefeng-Zou/SRPO_CODE
