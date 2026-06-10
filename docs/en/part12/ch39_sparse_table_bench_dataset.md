# Chapter 39: SparseTable-Bench — Data Engineering for Sparse Table Structure Robustness

## Abstract

This chapter uses SparseTable-Bench (STB) as a case study to examine the data engineering design of a sparse table structure recognition dataset. The difficulty of table understanding tasks lies not only in OCR text recognition, but also in the consistent representation of row-column topology, empty cells, spanning relationships, and spatial bounding boxes. The chapter first explains why conventional table recognition benchmarks struggle to cover sparse layouts and occlusion scenarios, then systematically describes STB's task boundaries, sample schema, synchronized HTML/text/bbox representations, and four-stage construction pipeline. The chapter further discusses the STB-Mask-Stress pressure test, TEDS/TEDS-S metric interpretation, training reproducibility interfaces, and error attribution methods. Through this case study, the chapter clearly demonstrates how a specialized dataset can simultaneously codify structural constraints, geometric supervision, and evaluation protocols into auditable data assets.

**Keywords**: table structure recognition; SparseTable-Bench; sparse tables; empty cells; TEDS; geometric alignment

**Learning Objectives**

- Understand the task boundaries between table structure recognition and conventional OCR and text extraction.
- Master schema design methods for synchronized representation of HTML, cell text, and bounding boxes.
- Explain the impact of empty cells, sparse layouts, and occlusion pressure tests on model evaluation.
- Distinguish the roles of TEDS, TEDS-S, and error interpretation in structural robustness evaluation.
- Design reproducible workflows for table data training, evaluation, and error attribution.

Tables are among the most underestimated objects in document intelligence. For ordinary paragraphs, a model that restores the reading order of text can typically satisfy the basic requirements of retrieval, summarization, or question answering. For tables, however, textual content is only surface-level information; the semantics are truly determined by the row-column topology, cell boundaries, spanning relationships, and the structural positions implied by blank regions. A cell with no text may still represent a valid column slot, a missing observation, or an alignment anchor relevant to its context. If a model skips these blank positions when generating HTML or Markdown, subsequent numerical comparisons, field extraction, and evidence localization all suffer cascading offsets.

SparseTable-Bench (STB) is a table structure robustness dataset built around precisely this problem. It targets not general OCR, nor table content recognition that only verifies whether cell text is correctly read, but rather the more specific task of Table Structure Recognition (TSR) and geometrically aware annotation. STB is composed of multi-source table images from scientific publications, financial reports, and clinical trial documents, with particular focus on borderless tables, sparse layouts, large blank regions, and complex row-column spans. The dataset provides each sample with a synchronized HTML structure sequence, cell-level text content, and fine-grained spatial bounding boxes. STB-Mask-Stress constructs a dedicated occlusion pressure test to observe whether models can still recover stable row-column structures under severe information loss.

In the preceding chapter on invoice extraction, structural constraints arose primarily from business fields, invoice layouts, and inter-field logical consistency. In SparseTable-Bench, structural constraints arise from the table's own row-column topology, empty cells, and spatial boundaries. This chapter introduces SparseTable-Bench from a data engineering perspective. The focus is not on reiterating the network architecture of any particular table recognition model, but on explaining how a benchmark designed for sparse tables defines task boundaries, organizes sample schemas, preserves empty cells, constructs pressure tests, interprets TEDS/TEDS-S metrics, and ultimately produces data assets that are trainable, evaluable, and reproducible.

## Keywords

Table structure recognition; sparse tables; robustness evaluation; structural annotation; dataset governance

## 39.0 Learning Objectives

Upon completing this chapter, the reader should be able to:

- Explain why table structure recognition, compared with OCR and text extraction, must treat row-column topology, empty cells, and spanning relationships as first-class supervision targets.
- Master the schema design principles by which SparseTable-Bench organizes each table into three synchronized representation layers: the HTML logical layer, the cell text semantic layer, and the bbox geometric layer.
- Distinguish the training, development, standard generalization, and occlusion pressure-testing responsibilities borne by each of the four splits: STB-Train, STB-Val, STB-Standard-Test, and STB-Mask-Stress.
- Analyze how missing empty-cell placeholders and left-shifted text systematically undermine the discriminative power of conventional TSR evaluation, and use this analysis to design annotation constraints that preserve empty positions.
- Explain the differences between the TEDS and TEDS-S metrics in structural robustness evaluation, and attribute errors to incorrect content reading, incorrect structure ordering, and misaligned placement using cell-level bounding boxes.

## 39.1 Why Table Structure Recognition Needs a Specialized Benchmark

In visual document understanding, tables are frequently reduced to "collections of text blocks in an image." This simplification can sometimes work for dense tables, because every row-column position has salient text that can serve as an implicit anchor. In scenarios with weak borders, large whitespace, many empty columns, or frequent cell merges, however, text anchors quickly fail. The model sees a set of discrete characters and large blank areas; the actual grid lines are not always visible, and the logical structure can only be inferred from visual arrangement, local alignment, and contextual priors.

Conventional TSR evaluation focuses mainly on whether the final HTML tree approximates the reference structure, or whether cell text is correctly populated. The problem is that if annotations do not explicitly preserve empty cells, a model that skips empty columns may still produce a seemingly reasonable text sequence; if evaluation only compares non-empty text, structural misalignment may be concealed; and if there are no cell-level bounding boxes, a model that generates the correct number of `<td>` nodes cannot be shown to have aligned those nodes with their true positions in the image. In domains such as financial statements, clinical trial tables, and experimental tables in papers, these errors are not formatting defects — they are structural errors that alter the field ownership of numerical values and the chain of evidence.

SparseTable-Bench addresses these risks at the data design stage. It treats each table sample as a composite structure composed of three layers of objects:

- **Logical layer**: row, column, cell, and spanning relationships expressed through HTML tags.
- **Semantic layer**: human-readable information expressed through the text content of each cell, with empty cells also preserved as explicit objects.
- **Geometric layer**: the spatial position of each cell in the image expressed through two-dimensional bounding boxes.

All three layers must coexist. Without bounding boxes alongside the HTML, the dataset cannot constrain whether the model has truly learned spatial alignment. Without HTML alongside the bounding boxes, the model cannot learn row-column topology and spanning structures. Without empty-cell placeholders alongside the non-empty text, evaluation will systematically underestimate the structural difficulty introduced by sparse layouts.

## 39.2 Dataset Overview and Task Boundaries

The task of SparseTable-Bench can be summarized as: "given a table image, recover a parseable structured representation while maintaining consistency among text, structure, and geometric positions." During training and evaluation, the model must output an HTML-like structure sequence while aligning text content and bounding boxes at the cell level. The dataset can be positioned as the data foundation for a spatially alignment-aware table recognition framework — usable for ordinary TSR training as well as for testing the structural robustness of VLMs under sparse, borderless, and partially occluded conditions. In related model experiments, STB often serves as the data basis for methods such as SA-Table and its Structural Prior Injection Adapter (SPIA). This chapter does not elaborate on the model architectures of SA-Table or SPIA; their requirements are cited only to motivate the data design decisions behind STB.

The public dataset entry point is:

https://huggingface.co/datasets/champion666/SparseTable_Bench_Dataset

In terms of scale, dataset documentation typically summarizes STB as "approximately 11,000 table images"; by split-level counts, the precise sample count is 10,983. To avoid ambiguity, this chapter uses the numbers from the split table as the authoritative reference.

| Split | Image Count | Annotation Format | Primary Purpose |
|---|---:|---|---|
| STB-Train | 8,000 | HTML + cell bbox | Multi-task supervised training |
| STB-Val | 1,000 | HTML + cell bbox | Hyperparameter selection and development-set evaluation |
| STB-Standard-Test | 1,000 | HTML + cell bbox | Standard generalization evaluation |
| STB-Mask-Stress | 983 | Occluded tables + topology labels | Robustness evaluation under sparse and information-deficient conditions |
| Total | 10,983 | — | Training, validation, standard testing, and pressure testing |

From the perspective of task boundaries, STB covers at least three categories of capability.

The first is table structure recognition. The model must recover structural tags such as `<table>`, `<tr>`, and `<td>`, and handle row-column organization, empty positions, merged cells, and local alignment. The emphasis here is not on reading out all the text in the image, but on placing that text back into the correct grid positions.

The second is geometrically aware annotation. Every cell has a bounding box `bbox=[x1,y1,x2,y2]`. This enables the dataset to be used for training position-aware output heads, for examining the correspondence between visual features and logical nodes, and for determining during error analysis whether the model read the content incorrectly, ordered the structure incorrectly, or placed correct content in the wrong position.

The third is mask pressure testing. STB-Mask-Stress actively reduces text cues through column-level and local occlusion, simulating severely sparse or incomplete visual conditions. Its objective is not to produce more difficult OCR, but to test whether the model can still preserve table topology — particularly empty columns, empty cells, and cross-column relationships — when portions of the content are invisible.

## 39.3 Sample Schema: Synchronized Representation of HTML, Text, and Bounding Boxes

A core design principle of SparseTable-Bench is to represent each table image as a synchronized multi-signal sample, rather than retaining only a single target format. The example below illustrates a simplified sample in which the second cell is empty, yet it remains a structurally valid column slot.

```json
{
  "html": "<table><tr><td>Revenue</td><td></td><td>$12.4M</td></tr></table>",
  "cells": [
    {
      "text": "Revenue",
      "bbox": [34, 52, 118, 74]
    },
    {
      "text": "[EMPTY_CELL]",
      "bbox": [118, 52, 215, 74]
    },
    {
      "text": "$12.4M",
      "bbox": [215, 52, 310, 74]
    }
  ]
}
```

The `[EMPTY_CELL]` token here is not ordinary text; it is a placeholder expressing "structure exists, content is absent." It decouples a cell's structural identity from its semantic content: even if the corresponding image region contains no readable characters, that position still has row-column coordinates, a bounding box, and contextual relationships. For sparse tables, this placeholder prevents the model from treating blank regions as non-existent during generation, thereby reducing the probability of column collapse and left-shift errors. Figure 39-2 summarizes the synchronized relationship among the three supervision signals — HTML, text, and bounding boxes — within the same table sample.

![Figure 39-2: Three synchronized supervision signals in a table sample](../../images/part12/ch39_02_supervision_schema_en.png)

From a data engineering perspective, the sample schema of STB includes at least the following fields and validation rules.

| Object | Typical Fields | Purpose | Key Quality Checks |
|---|---|---|---|
| Image | `image_id`, image file, width/height | Serves as visual input and bbox coordinate reference | Image opens successfully; resolution is consistent with bbox coordinate system; no corrupted pages |
| HTML structure | `html`, `rowspan`, `colspan` | Expresses logical topology and output sequence | HTML parses correctly; row and column counts are consistent; merged cells do not cause grid conflicts |
| Cell text | `cells[i].text` | Expresses the semantic content of each cell | Text order matches HTML order; empty cells use a uniform placeholder; special characters are normalized |
| Empty cells | `[EMPTY_CELL]` or equivalent empty-position marker | Preserves positions that are structurally valid but textually empty | Not filtered out due to empty text; bbox still present; participates in structural evaluation |
| Spatial bounding boxes | `cells[i].bbox=[x1,y1,x2,y2]` | Constrains the alignment between visual regions and structural nodes | Coordinates within bounds; positive area; approximate row-column alignment; one-to-one correspondence with cells |
| Data split | `split`, version number, source domain | Supports training, validation, and reproducible evaluation | No train/test leakage; relationship between pressure and standard sets is clear; version is traceable |

The value of this schema lies in allowing the same data to serve multiple training and evaluation objectives. If the model is a purely generative VLM, image-to-HTML sequence supervision can be used; if the model includes a position prediction head, bbox regression or discrete coordinate tokens can be used; if the research goal is empty-cell recovery, the recall rate of `[EMPTY_CELL]`, the empty-column preservation rate, and structural edit distance can be tracked specifically. The closer a data asset is to this "multi-signal synchronized" form, the easier it becomes to perform error attribution in model experiments.

It is important to note that the specific notation for the empty-cell token must remain consistent across the dataset, tokenizer, training scripts, and evaluation scripts. The dataset uses `[EMPTY_CELL]` to represent empty cells; in some model paper contexts, typographic variants such as `[EMPTY CELL]` may also appear. During engineering implementation, one canonical form should be selected and enforced uniformly during the data transformation stage; otherwise, the same empty position may be tokenized differently, causing the training objective and the evaluation objective to become misaligned.

## 39.4 Four-Stage Construction Pipeline

The construction of SparseTable-Bench can be organized into four stages: table collection, structure extraction, spatial annotation, and sparse topology augmentation. These four stages are not a simple serial file transformation; rather, they involve repeated validation of consistency among structure, text, and geometry, as illustrated in Figure 39-1.

![Figure 39-1: Four-stage SparseTable-Bench construction pipeline](../../images/part12/ch39_01_stb_pipeline_en.png)

### 39.4.1 Table Collection

Raw table images are sourced from multi-source documents including scientific publications, financial reports, and clinical trial documents. These sources are chosen because they naturally contain large numbers of irregular tables: scientific papers frequently feature borderless experimental results tables and meta-analysis tables; financial reports commonly contain multi-level headers, blank groupings, and cross-column annotations; clinical trial documents routinely mix metrics, groups, time points, and missing observations within a single table. Compared with templated invoices or fixed-format forms, these tables are more likely to expose VLM dependencies on implicit structure.

The key at the collection stage is not to blindly increase the number of images, but to cover diversity in sparse structure. Data engineers must focus on at least four types of samples: tables lacking borders but with clearly aligned rows and columns; tables with large blank areas or many empty columns; tables containing complex `rowspan`/`colspan` relationships; and tables in which text density varies greatly across different regions. These samples constitute the foundation that distinguishes STB from ordinary dense-table datasets.

### 39.4.2 Structure Extraction

The structure extraction stage converts a table's logical topology into an HTML sequence. HTML is not the only viable format, but it offers two advantages: its tag tree naturally accommodates the hierarchical expression of rows, columns, and cells; and mainstream table structure metrics such as TEDS can be computed directly on HTML trees. For ordinary cells, annotations must specify the row and column to which each cell belongs; for merged cells, `rowspan` and `colspan` must be preserved; for empty cells, the corresponding `<td>` nodes must be retained rather than deleted due to the absence of text.

The most common error at this stage is "visually plausible but logically unparseable grids." For example, if a single empty `<td>` is missing from one row, a human reviewer may not notice, but after converting the HTML tree to a matrix, every subsequent column index in that row will be shifted left. Structure extraction therefore cannot rely solely on manual visual inspection; a parser should also be used to convert the HTML back into a grid matrix, checking the number of columns after each row is expanded, the coverage areas of merged cells, the count of empty positions, and the ordering of cells.

### 39.4.3 Spatial Annotation

The spatial annotation stage assigns two-dimensional bounding boxes to each cell. Bounding boxes are not merely auxiliary visualization fields; they determine whether the dataset can train and evaluate geometric alignment capability. For cells with text, the bounding box should cover the cell region rather than only the text region; for empty cells, bounding boxes must still be inferred from neighboring row-column boundaries, the overall table layout, and implicit grid structure. This allows the model to learn the structural prior that "a region without text may still be a valid cell."

Quality checks can be divided into geometric validity and topological consistency. Geometric validity covers coordinates within bounds, positive width and height, and bounding box dimensions consistent with image size. Topological consistency covers the requirement that cells in the same row have substantially overlapping vertical extents, cells in the same column have substantially aligned horizontal extents, and bounding boxes for merged cells cover the corresponding row-column areas. For sparse tables, topological consistency is often more important than text OCR, because large blank regions cannot be verified through text.

### 39.4.4 Sparse Topology Augmentation

The sparse topology augmentation stage is used to construct pressure tests and supplement robustness signals. Rather than simply randomly occluding the image, it applies controlled masking based on column, header, body, and cell topology. After occlusion, the corresponding regions in the image are filled with a uniform background color, and the text tokens in the annotations are simultaneously set to empty or removed, but the cell nodes, row-column positions, and topological relationships are preserved. Samples constructed this way reduce the model's reliance on local text cues, forcing it to use remaining layout, adjacent cells, and structural priors to recover the table.

The construction pipeline should ultimately produce three types of auditable artifacts: standard training/validation/test samples, STB-Mask-Stress pressure test samples, and data documentation recording the data version, source domains, split strategies, and transformation script hashes. Without this metadata, a benchmark easily becomes single-use experimental material, unable to support subsequent model iteration and cross-study comparison.

## 39.5 How Empty Cells and Sparse Layouts Induce Structural Errors

The core difficulty of sparse tables is not simply "too much whitespace, so too little information." Whitespace itself carries structural meaning. A blank region may represent an empty cell, an entire column of missing values, the area occupied by a spanning cell, or merely typographic whitespace on the page. If a model cannot distinguish among these cases, structural hallucinations will occur.

One typical error is empty-column skipping. Suppose a table truly has three columns, and the middle column is mostly empty. During generation, the model may output only the first and third columns, moving the content of the third column into the second-column position. At the text level, the major values have all been recognized; at the structural level, the column semantics have changed. In a financial statement, this may cause "current-period figures" to be interpreted as "prior-period figures"; in a clinical table, it may cause treatment-group metrics to be misassigned to the control group.

Another error is cascading misalignment. Table recognition typically generates output autoregressively; if a single empty `<td>` is missed earlier, all subsequent cells in that row shift left. If this error occurs in a multi-level header, the effect extends to multiple rows and multiple fields. Conventional average scores may show only a slight drop, but the actual business meaning has been completely distorted.

These structural errors are further reflected in evaluation results. In sparse tables, missing empty positions, column shifts, and erroneous filling of empty cells change the number of HTML nodes, node order, and row-column expansion relationships. They therefore usually reduce TEDS or TEDS-S. STB preserves HTML, text, and bounding boxes simultaneously so that score drops can be traced back to inspectable data objects rather than remaining only at the aggregate-metric level.

Errors in sparse tables can be classified into five types.

| Error Type | Manifestation | Primary Cause | Observation Method in STB |
|---|---|---|---|
| Missing empty position | Empty `<td>` not generated; column count decreases | Empty cells lack visual text anchors | `[EMPTY_CELL]` recall, TEDS-S, row-column expansion check |
| Column left-shift / right-shift | Non-empty content placed in adjacent column | Intermediate empty column skipped or merged | HTML matrix alignment, bbox-to-column-index consistency |
| Empty-cell filling | Model generates nonexistent text in empty positions | Contextual completion or overly strong language prior | `[EMPTY_CELL]` precision, cell-text inspection |
| Incorrect merging relationships | `rowspan`/`colspan` missing or mislabeled | Sparse region boundaries are weak; header levels are complex | Structure tree edit, merge-area coverage check |
| Spatial drift | HTML structure parses correctly but bboxes do not align | Model learned sequences only, lacking geometric supervision | Cell bbox IoU, row-column geometric alignment check |

These errors demonstrate that the value of SparseTable-Bench is not simply providing a more difficult dataset, but transforming structural failure modes in sparse tables into supervision objects that are annotatable, computable, and attributable.

## 39.6 STB-Mask-Stress: A Pressure Test for Information-Deficient Conditions

STB-Mask-Stress is the robustness evaluation split within SparseTable-Bench, dedicated specifically to pressure testing. Its design philosophy is to systematically reduce text cues — while preserving table topology — and to observe whether the model can still recover row-column structure and empty cell positions. Unlike ordinary data augmentation, the goal of STB-Mask-Stress is not to increase training set diversity, but to construct an evaluation environment that more closely resembles a "structural understanding stress test." This chapter follows the dataset documentation in using the name STB-Mask-Stress; in related experimental contexts, it can also be understood as a masked table evaluation setting oriented toward column-level occlusion, suitable for use with pressure-test metrics such as Masked-TEDS.

Figure 39-3 illustrates the basic workflow of STB-Mask-Stress, from column-level occlusion generation to evaluation interpretation.

![Figure 39-3: STB-Mask-Stress occlusion generation and evaluation workflow](../../images/part12/ch39_03_mask_stress_flow_en.png)

The occlusion strategy of STB-Mask-Stress is column-aware. The workflow can be summarized as follows.

1. Parse the original table structure to obtain the row-column index, header/body membership, and bounding box of each cell.
2. Randomly select a subset of columns as occlusion candidates.
3. Sample an occlusion pattern for each selected column. If the sampled probability falls in the body masking range, occlude the body cells in that column while preserving the header; if it falls in the header masking range, occlude the header cells while preserving the body; otherwise, randomly occlude a subset of body cells to produce intermittent blanks.
4. Fill the selected cell regions in the image with a uniform background color.
5. Synchronously update annotations: text tokens in occluded regions are removed from or set to empty in the target, but row-column topology is retained.
6. Compute TEDS, TEDS-S, or masked versions of structural metrics on the updated samples, and perform error attribution.

The three occlusion patterns assess different capabilities. Body Masking retains headers but removes body content, testing whether the model can maintain column positions based on column headers and geometric structure. Header Masking removes headers, testing whether the model can maintain body alignment when column semantics are absent. Random Sparse Masking produces local breaks and intermittent blanks, more closely approximating sparse conditions caused by real-world scanning artifacts, occlusions, or rendering defects.

It must be emphasized that STB-Mask-Stress scores should not be equated directly with standard test set generalization scores. The standard test set measures the model's overall recognition ability on natural tables; the pressure test measures the model's structural recovery ability under information-deficient and visually sparse conditions. A model with a high TEDS on the standard test but a noticeably lower TEDS-S on STB-Mask-Stress likely depends on visible text anchors rather than having stably learned row-column topology. Conversely, a model with a stable structural score on the pressure test but a dropping text score may have preserved the topology while being unable to recover occluded content — this represents a different type of capability boundary.

From a data engineering perspective, the key to pressure testing is "occlusion and annotation synchronization." If images are occluded without updating labels, the model is required to predict invisible text, and evaluation results will conflate language memorization and guessing ability. If text is removed while also deleting cell nodes, the pressure test degenerates into an ordinary sparse table, making it impossible to test empty-position preservation. STB-Mask-Stress should therefore always uphold one principle: text evidence may be removed; structural topology must not be arbitrarily eliminated.

## 39.7 Evaluation Protocol: TEDS, TEDS-S, and Error Interpretation

SparseTable-Bench uses Tree-Edit-Distance-based Similarity (TEDS) and its structural variant TEDS-S as primary evaluation metrics. TEDS parses both the predicted HTML and the reference HTML into trees, computing a normalized tree-edit similarity. It is jointly influenced by structural tags, node ordering, and cell text content. TEDS-S ignores text content and focuses more on structural topology — for example, row-column alignment, merged cell recovery, and empty cell positions.

These two metrics are appropriate for cross-model comparison, but must not be interpreted mechanically. Especially for sparse table datasets such as STB, metric differences often correspond to different error sources.

In extremely sparse tables or tables with many empty cells, lower TEDS/TEDS-S scores usually originate from structural prediction errors. Numerous empty positions weaken visible text anchors. A model may fill nonexistent content into empty cells, skip empty columns, or assign neighboring-column content to the wrong column position. Once these errors enter the HTML output, the number of `<td>` nodes, node order, and row-column expansion relationships change, ultimately reducing TEDS or TEDS-S. TEDS-S further focuses on structural topology and empty-cell positions, making it especially useful for exposing such row-column alignment errors.

| Metric Pattern | Possible Interpretation | Conclusion That Should Not Be Drawn | Supplementary Check |
|---|---|---|---|
| TEDS high, TEDS-S high | Structure and text are broadly stable | Does not imply bboxes are necessarily correct | Cell bbox IoU, row-column geometric alignment |
| TEDS low, TEDS-S high | Structure is largely correct but text content is wrong | Does not imply the model has poor structure recognition | OCR/text normalization, number formatting |
| TEDS-S low, TEDS close or slightly higher | Some text is correct but structural misalignment exists | Cannot rely on text match rate alone | Empty-cell recall, column-shift inspection |
| Standard set high, Mask-Stress low | Depends on visible text anchors; poor resistance to sparseness | Does not imply the model is unusable in ordinary scenarios | Statistics broken down by occlusion pattern: body/header/random |
| Mask-Stress TEDS-S high but TEDS low | Topology well preserved, occluded text unrecoverable | Cannot require the model to recover invisible content from nothing | Confirm that occluded text has been synchronously removed from targets |

When interpreting TEDS/TEDS-S in STB, three types of information should be considered together.

First, TEDS is a mixed metric of structure and text. It is appropriate for assessing how closely the final HTML output approximates the reference, but when scores drop, it is necessary to disaggregate the causes — text misrecognition, tag-tree errors, or cell ordering misalignment. For sparse tables, the same TEDS decrease may represent entirely different risks.

Second, TEDS-S is a structural metric, but not a geometric metric. By ignoring text, it more clearly reflects row-column topology, but it is still tree-based and does not directly verify whether bounding boxes correspond to image positions. If a model outputs a topologically correct HTML structure but places cell bounding boxes in incorrect visual regions, TEDS-S will not adequately penalize this. For geometrically aware models using STB, additional checks such as bbox IoU, centroid distance, row-column alignment error, or cell-to-region assignment should be added.

Third, pressure test scores should be reported by occlusion type. Body Masking, Header Masking, and Random Sparse Masking assess different capabilities. Reporting only a single average score may conceal differences such as the model collapsing when headers are absent, remaining stable under body occlusion, and drifting under random sparseness. Data engineering practice typically reports the overall score together with per-mask-type scores and representative failure cases.

In addition to the primary metrics, STB is well suited to introducing several diagnostic metrics. For example, empty-cell recall measures whether `[EMPTY_CELL]` positions are preserved; column-count expansion consistency measures whether each row, when expanded, matches the reference column count; merged-cell accuracy measures `rowspan`/`colspan` correctness; and bbox match rate measures whether structural nodes correspond to their visual regions. These metrics need not all appear in leaderboards, but they are highly suited for model debugging and data quality inspection.

## 39.8 Data Engineering Practice: Using STB for Training and Reproduction

When using SparseTable-Bench for model training, the most common approach is to use images as input and organize HTML structure, cell text, and bounding boxes into a unified output sequence or multi-task supervision target. For generative VLMs, the model can directly generate HTML and insert text or empty-cell tokens at the cell content positions; for models with position heads, bbox regression or coordinate token prediction can be added alongside text generation; for adapter-based or structural prior models, bounding boxes and grid topology can be converted into auxiliary structural features to help the decoder maintain row-column alignment during generation.

Several constraints are easily overlooked in practice.

First, data transformation must maintain consistent sample ordering. The `i`-th cell in the HTML, `cells[i].text`, and `cells[i].bbox` must refer to the same logical cell. If a transformation script alters the ordering while filtering empty text, expanding merged cells, or sorting bounding boxes, the training target becomes noise.

Second, empty cells must not be deleted during the cleaning stage. Many general-purpose document cleaning scripts filter out empty strings, empty tags, and empty bounding boxes as invalid fields. In STB, these are precisely the core supervision signals that must be preserved. Cleaning rules must therefore distinguish between "invalid missing data" and "valid empty cells."

Third, bounding box coordinates require an explicit coordinate system. Different models may use original pixel coordinates, normalized coordinates, or discrete token coordinates. Transformations should record image width and height, scale factors, and padding strategies to prevent training and evaluation from using different coordinate systems.

Fourth, standard test results and pressure test results must be reported separately. If STB-Mask-Stress is mixed into a general test set average, reviewers will have difficulty determining whether the model's shortfall reflects insufficient generalization on natural scenarios or insufficient robustness in extremely sparse scenarios. A clearer reporting format is: first present TEDS/TEDS-S on the Standard Test, then present TEDS/TEDS-S on the Mask-Stress split, broken down by occlusion type.

Fifth, error cases should be traced back to data objects. A single failure can be decomposed into several questions: Is the HTML parseable? Is the row-column expansion consistent? Were any empty cells missed? Was any text misread? Were any bounding boxes offset? Only by doing so do model fix actions become concrete: whether more empty-cell samples are needed, position supervision needs improvement, the tokenizer needs correction, or annotations need re-validation.

For reproducibility, using STB should not stop at the coarse-grained procedure of "load data, train model, report score." A more rigorous approach is to decompose each experiment into four auditable stages. Stage one is data loading verification: randomly sample training, validation, standard test, and pressure test samples to confirm that the image, HTML, cell list, and bounding boxes can all be associated through a single sample ID. Stage two is schema rendering verification: expand the HTML into a two-dimensional grid and overlay bounding boxes on the original image to confirm that empty cells, merged cells, and non-empty text are visually interpretable. Stage three is model input-output verification: clarify whether the model receives the original image, a cropped image, or a patch-based image, and whether it outputs pure HTML, HTML with coordinate tokens, or multi-task HTML and bbox results. Stage four is evaluation and attribution verification: compute Standard-Test and STB-Mask-Stress scores separately, then sample and review failures by the four error categories of empty-position miss, column shift, text error, and spatial drift.

| Reproduction Stage | Input Objects | Output Objects | Key Checks |
|---|---|---|---|
| Data loading | Image, HTML, cells, bbox | Unified sample record | ID alignment, field completeness, correct split assignment |
| Schema rendering | HTML tree, cell list | Two-dimensional grid and visualization overlay | Empty cells preserved, merge relationships parseable, bboxes within bounds |
| Model training | Table images and multi-task labels | HTML sequence, text tokens, bbox predictions | `[EMPTY_CELL]` vocabulary consistency, coordinate system consistency, reasonable loss masking |
| Evaluation attribution | Predicted results and reference annotations | TEDS, TEDS-S, diagnostic error table | Standard set and pressure set reported separately; errors traceable to data fields |

For VLM or document model training, STB can serve two different roles. As training data, it provides structured visual supervision suited to helping models learn the alignment from "visual regions to logical cells." As evaluation data, it is better positioned as a robustness slice for verifying whether models depend solely on text density and local OCR cues. If a general-purpose VLM performs well on natural image QA but cannot preserve empty columns on STB-Mask-Stress, its document structural capability still requires specialized data reinforcement. If a document model has high structural scores on the standard test but exhibits extensive spatial drift in bbox verification, the model may have learned HTML language patterns without having genuinely established geometric alignment ability.

Therefore, in coursework experiments or project work, STB evaluation results are typically decomposed into two layers: "usability" and "robustness." The usability layer examines table structure recovery quality on the standard test set, answering whether the model can handle typical scientific, financial, and clinical tables. The robustness layer examines empty cells, occluded columns, and locally sparse regions in the pressure test, answering whether the model can maintain a credible structure when evidence is absent. This layered reporting approach is more suitable for data engineering retrospectives than a single leaderboard score, and more readily supports the flow of failure samples back into cleaning, annotation, and retraining pipelines.

In team collaboration, STB should also be elevated from "experimental data" to "deliverable data asset." A deliverable version should include at minimum three types of records. The first type is a data card recording data provenance, license status, sample size, split methodology, field schema, empty-cell conventions, and the bounding box coordinate system. The second type is an evaluation card recording the model used, input resolution, decoding parameters, TEDS/TEDS-S computation script version, whether OCR post-processing is enabled, and the occlusion strategy for Mask-Stress. The third type is an error card recording representative failure samples, error types, whether caused by annotation issues, whether caused by model output issues, and the next round of corrective actions. Without these records, even if scores are reproducible, the causes of failures are difficult to audit.

In particular, errors related to empty cells should not appear only as a few case screenshots in the final report. A better practice is to consolidate them into queryable error slices — for example, "entire column empty but column header visible," "column header empty but body dense," "blank region spanning multiple rows," and "empty cell adjacent to merged cell." Each slice can independently report TEDS-S, empty-cell recall rate, and column-expansion consistency rate. This way, when a new model version improves on the average score but regresses on the empty-column slice, the data team can promptly detect the robustness regression rather than waiting until incorrect column interpretation surfaces on the business side.

Annotation quality inspection can also adopt a dual-channel approach. The first channel checks structure: whether the HTML can be stably parsed into a two-dimensional matrix, whether the row and column counts are consistent after expansion, and whether `rowspan` and `colspan` produce overlaps or holes. The second channel checks geometry: whether bounding boxes are within bounds, whether they cover the cell region, whether the horizontal extents of cells in the same column are continuous, and whether bounding boxes for empty cells together with adjacent cells form a reasonable grid. Only when both structure and geometry pass should a sample enter the training set; if text is correct but structure or bounding boxes are suspect, the sample should enter a rework queue rather than be used directly as a supervision signal.

A reproducible benchmark version requires pinned version numbers for the training set, validation set, standard test set, and STB-Mask-Stress, along with retained hash values of data generation scripts. Pressure tests especially require versioning, because small changes to the number of occluded columns, occlusion probability, or background fill value all affect model scores. If the occlusion strategy is adjusted in the future, it should be released as a new pressure test version rather than overwriting existing results. Only then can STB support long-term model iteration, cross-team comparison, and in-book project reproducibility experiments.

## 39.9 MindSpore Implementation and Code

To facilitate experimental reproduction and review of data processing workflows, the MindSpore companion implementation entry point for SparseTable-Bench is:

https://github.com/champion666/SparseTable-Bench-MindSpore

This repository serves as the companion implementation entry for this chapter, organizing data reading, occlusion construction, and evaluation reproduction experiments. A complete companion implementation should typically include: an STB data reader; HTML and cell schema transformation scripts; `[EMPTY_CELL]` token normalization; bounding box coordinate transformation; an STB-Mask-Stress generation script; TEDS/TEDS-S evaluation scripts; and a minimal example configuration for MindSpore training. Only then can a closed loop be formed among the book chapter, dataset documentation, and code repository.

In addition to the GitHub code entry, the public dataset address should cross-reference the code repository in the README:

https://huggingface.co/datasets/champion666/SparseTable_Bench_Dataset

It is important to note that the role of the code repository is not to simply replicate paper experiments, but to support reproduction of the data engineering workflow described in this chapter: loading samples, validating the schema, constructing occlusions, running evaluation, and interpreting errors. As long as these interfaces remain stable, subsequent substitution of SA-Table, OCRFlux, Qwen-VL, or any other table recognition model can be compared under the same data protocol.

## 39.10 Connections to Adjacent Chapters

SparseTable-Bench connects naturally to multiple parts of this book.

With respect to the document understanding and cross-modal alignment topics in Part III, STB provides a stricter example than ordinary OCR: visual regions, text content, and structure tokens must be simultaneously aligned. The OCR and document structure re-annotation discussed in Chapter 9 is concretely instantiated here as the synchronization of cell-level text, bounding boxes, and HTML. The cross-modal alignment discussed in Chapter 11 is concretely instantiated here as the alignment of table image regions with logical cell nodes. This chapter can therefore be viewed as a specialized case study advancing from "page text recognition" toward "structured visual object recovery."

Compared with Chapter 38 on invoice document understanding, StructBill-CN places greater emphasis on business schemas, field extraction, and logical consistency, while SparseTable-Bench places greater emphasis on intra-table topology, empty cells, and sparse layouts. Both belong to visual document data engineering, but one targets high-risk invoice fields and the other targets general table structure robustness.

Compared with Chapter 40 on multi-chart infographic reasoning, STB focuses on structural recovery within a single table object, while multi-chart infographic reasoning focuses on cross-chart evidence aggregation and multi-step computation. The former provides foundational capability for the latter: if a model cannot stably recover column positions within a single table, numerical reading and evidence localization in cross-chart reasoning will lose their reliable basis.

Looking ahead, STB connects directly to Chapter 47 on VLM data recipes. Chapter 47 examines how multimodal training data organizes images, text, coordinates, and instruction signals; STB provides exactly such a structured visual supervision example: the input is a table image, and the output simultaneously includes HTML structure, cell text, bounding boxes, and empty-cell placeholders. It can serve as a document-table slice in VLM data recipes, demonstrating why general image-caption pairs are insufficient for training stable table structure capability.

In the Part XIV projects, STB can also connect to P03 and P05. The LLaVA multimodal instruction data factory in P03 needs to convert document images into trainable visual instruction samples; STB can provide instruction sources such as "identify the table structure," "indicate the positions of empty cells," and "explain column-position shift errors." The multimodal RAG project in P05 needs to extract retrievable evidence from PDFs, financial reports, and scientific publications; STB can help parse tables into structured evidence that is citable, comparable, and traceable. Particularly in financial reports, medical papers, and scientific publications, table structure errors are often harder to detect and more likely to affect final answers than individual OCR character errors.

## Chapter Summary

The core contribution of SparseTable-Bench is that it transforms the problem of structural robustness in sparse tables into a data engineering problem that is annotatable, trainable, and evaluable. The dataset establishes three categories of supervision signals through HTML structure sequences, cell-level text content, and fine-grained bounding boxes, and uses `[EMPTY_CELL]` to explicitly preserve empty-cell topology, preventing blank regions from being incorrectly removed during cleaning, training, and evaluation. STB-Mask-Stress further constructs pressure tests through column-aware occlusion, enabling the model's structural recovery capability under severe information deficiency to be observed in isolation.

When using this dataset, relying on a single overall TEDS score is insufficient. TEDS, TEDS-S, bounding box inspection, empty-cell recall, and per-mask-type error analysis should be used in combination to distinguish text errors, structural errors, and spatial errors. The value of STB is not simply providing a "harder" table dataset, but transforming the failure patterns of sparse tables into objects that are annotatable, trainable, and evaluable. For large model data engineering, the lesson of STB is this: the value of a complex document dataset comes not only from sample scale, but from whether real failure patterns have been encoded into the schema, the construction workflow, and the evaluation protocol.

## References

1. Zhong, X., ShafieiBavani, E., & Yepes, A. J. (2020). Image-based Table Recognition: Data, Model, and Evaluation. ECCV 2020.
2. Smock, B., Pesala, R., & Abraham, R. (2022). PubTables-1M: Towards Comprehensive Table Extraction From Unstructured Documents. CVPR 2022.
3. Zhu, F., Lei, W., Huang, Y., Wang, C., Zhang, S., Lv, J., Feng, F., & Chua, T.-S. (2021). TAT-QA: A Question Answering Benchmark on a Hybrid of Tabular and Textual Content in Finance. ACL 2021.
4. Pandas Development Team. (2026). pandas Documentation. https://pandas.pydata.org/docs/
5. Apache Arrow Contributors. (2026). Apache Arrow Documentation. https://arrow.apache.org/docs/
