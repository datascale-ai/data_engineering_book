# Chapter 22: Multimodal RAG and Visual Retrieval

## Abstract

Real-world knowledge does not always exist in plain text: trends in financial reports are embedded in line charts, the steps in a manual depend on button positions in screenshots, and the signatures and annotations in contracts exist as image regions. When Retrieval-Augmented Generation (RAG) expands from text-centric scenarios to such visually complex documents, simply flattening documents into character strings via Optical Character Recognition (OCR) discards visual structure, spatial relationships, and image-text alignment information, leading to systematic failures. This chapter frames the core task of multimodal RAG not as appending OCR to text RAG, but as fundamentally redefining knowledge units, indexing strategies, and evidence organization. The chapter first analyzes why text RAG cannot cover visual knowledge, then discusses how visual chunking and object modeling organize pages, regions, tables, and images into locatable and citable knowledge assets, before explaining the implementation of cross-modal indexing, retrieval, and reranking. It concludes by establishing evaluation, error attribution, and online failure-sample replenishment mechanisms for visual retrieval, providing a methodological foundation for applications such as complex enterprise document understanding and financial report analysis.


As RAG systems progressively expand from text-centric scenarios—enterprise knowledge Q&A, policy retrieval, and document assistants—to financial report analysis, contract review, product manual comprehension, invoice processing, medical imaging reports, and complex-layout document Q&A, a new problem has come to the fore: real-world knowledge does not always exist in plain text. A large volume of critical information is hidden in images, tables, charts, screenshots, flowcharts, layout structures, and visual regions. If the system still follows a text-RAG paradigm—parsing documents into character strings, then chunking, vectorizing, and retrieving—it will readily suffer systematic failures in complex knowledge scenarios.

The fundamental assumption of traditional text RAG is that knowledge can be converted into text and that Q&A can be completed through text retrieval and generation. In multimodal scenarios, however, this assumption does not always hold. For example, key trends in a financial report may be encoded in a line chart; the operating steps in a product manual may depend on button positions in a screenshot; signatures and annotations in a scanned contract may exist as image regions; and the basis for a judgment in a medical examination report may arise from the combined relationship between image regions and textual descriptions. In these cases, OCR can extract only part of the text and cannot fully express visual structure, spatial relationships, or image-text alignment information.

Therefore, the core task of multimodal RAG is not simply adding OCR to text RAG, but redefining knowledge units, indexing strategies, and evidence organization. The system must understand the relationships among pages, regions, objects, tables, images, and text, and organize this heterogeneous information into knowledge assets that are retrievable, locatable, citable, and verifiable. In other words, multimodal RAG does not address "how to convert images to text" but rather "how to bring visual knowledge into the retrieval-and-generation loop."

This chapter focuses on multimodal RAG and visual retrieval, emphasizing why text RAG cannot cover visual knowledge, how to design visual chunks and object models, how to implement cross-modal indexing, retrieval, and reranking, and how to continuously improve system capability in complex document scenarios through evaluation, error attribution, and online failure-sample replenishment. This chapter also provides the methodological foundation for subsequent projects on multimodal financial report assistants, complex enterprise document retrieval, and document understanding.

## Keywords

Multimodal RAG; Visual Retrieval; Visual Chunk; Cross-modal Indexing; Error Attribution; Data Replenishment

## Learning Objectives

- Be able to explain why text RAG and OCR cannot cover visual knowledge such as charts, layouts, and interfaces.
- Be able to distinguish the processing priorities for document layout knowledge, chart numerical knowledge, and interface/scene object knowledge.
- Be able to design visual chunk and object modeling methods that make pages, regions, tables, and images locatable and citable.
- Be able to construct cross-modal indexing, retrieval, and reranking pipelines to recall multimodal evidence.
- Be able to establish evaluation, error attribution, and online failure-sample replenishment mechanisms for visual retrieval.

---

## 22.1 Why Text RAG Cannot Cover Visual Knowledge

### 22.1.1 From Readable Text to Comprehensible Pages

In Chapter 21, we discussed how the RAG data pipeline revolves around document engineering: from raw document ingestion, parsing, and structured cleaning, to chunk construction, indexing, retrieval, evaluation, and feedback. For knowledge bases that are primarily text-based, this pipeline can already solve a large number of practical problems. However, when knowledge appears in the form of complex pages, charts, images, or layout structures, relying solely on text extraction encounters clear bottlenecks (Xu et al. 2020; Huang et al. 2022).

The core processing unit of text RAG is the text chunk. The system typically first parses a document into text, then segments it into chunks based on paragraph or semantic boundaries, and finally retrieves relevant content via vector search and keyword search. The implicit premise of this approach is that raw knowledge can be converted into linear text with reasonable completeness. However, knowledge in complex documents is often non-linear—it is spatial, structured, and cross-modal. For instance, in a financial report, information such as "revenue growth," "gross margin decline," and "cash flow changes" is typically not expressed within a single paragraph but is distributed across body text, tabular data, bar charts, line charts, and footnotes. Extracting only the body text may miss chart trends; extracting only tables may lose interpretive context; OCR-ing chart labels alone cannot capture curve trends. To answer the user question "What are the main reasons for the company's declining profit margin in Q2?", the system must simultaneously retrieve textual explanations, key financial tables, and relevant charts—not just a single text passage. Similarly, in a product operation manual, a step might say "click the Advanced Settings button in the upper right corner," but the actual button position, icon appearance, and interface layout exist in a screenshot. If the system extracts only text, it can obtain the phrase "Advanced Settings button" but cannot understand where the button is on the page, nor can it answer questions like "In which area is the Advanced Settings button?" or "What other options are next to that button?" The answers to these questions reside in the visual layout, not in text alone.

Therefore, multimodal RAG first requires a perspective shift from "readable text" to "comprehensible pages." A page is no longer merely a container for text but a knowledge structure composed of text blocks, image blocks, table blocks, chart regions, layout relationships, and visual objects. What the system must process includes not only textual content but also the spatial relationships, reference relationships, and semantic relationships among these elements.

---

### 22.1.2 OCR Is Not Visual Understanding

In multimodal RAG projects, a common misconception is treating OCR as a substitute for visual understanding. The role of OCR is to recognize the text within images; it answers the question "What text is in the image?" Visual understanding, however, must address more complex questions: Where is that text located? Which visual elements is it associated with? Which table, chart, or region does it belong to? What relationships exist among these elements? And how do they jointly support an answer? (Kim et al. 2022; Mathew et al. 2021). Consider a scanned financial report: OCR can recognize text and numbers on the page—for example, "Revenue," "Gross Margin," "2023," "2024," "15.6%"—but having these strings is insufficient to answer questions. The system must also know which row and column a given number belongs to, what its unit is, whether it comes from a merged cell, whether it is subject to footnote constraints, and whether it corresponds to a trend in a specific chart. If OCR output is naively concatenated into text, the original visual structure is destroyed and the model sees only a jumbled string.

Similarly, for chart-based knowledge, OCR can recognize axis labels and legend text, but it is difficult to directly understand curve trends, bar comparisons, area changes, or outliers. If a user asks "Which quarter had the fastest revenue growth?", the answer may not be text that appears explicitly on the page but must be inferred from the shape of the chart. In this case, the system needs visual region detection, chart type identification, data extraction, and cross-modal alignment capabilities—not just OCR.

For screenshot-based knowledge, OCR is equally insufficient. The buttons, menus, input fields, icons, and tooltips in a software interface collectively constitute operational semantics. Whether a button is clickable, which region it belongs to, and which input field it is associated with are typically determined by visual layout. OCR can recognize button labels but cannot fully express interaction relationships. If the system relies solely on OCR, it may report the button name but cannot guide the user through the operation.

Therefore, in multimodal RAG, OCR is only a foundational capability, not a complete solution. True visual understanding must simultaneously handle four categories of information: textual content, visual regions, spatial layout, and cross-modal relationships. Only when this information is organized into a unified knowledge representation can the system stably answer questions in complex document and visual scenarios.

![Figure 22-1: The Capability Boundary Between OCR and Visual Understanding](../../images/part7/图22_1zh.png)

*Figure 22-1: The Capability Boundary Between OCR and Visual Understanding*



---

### 22.1.3 Three Canonical Forms of Visual Knowledge

To understand the data engineering challenges of multimodal RAG, and drawing reference from benchmarks ranging from DocVQA and InfographicVQA to ChartQA (Mathew et al. 2021; Mathew et al. 2022; Masry et al. 2022), we can first classify visual knowledge into three categories: document layout knowledge, chart numerical knowledge, and interface/scene object knowledge. Different types of visual knowledge impose different requirements on parsing, chunking, indexing, and evaluation.

The first category is document layout knowledge. It exists primarily in PDFs, scanned documents, contracts, reports, manuals, and invoices. Its defining characteristic is that knowledge depends on page layout and structural hierarchy. For example, the relationships between headings and body text, between tables and footnotes, between figure captions and images, and between the phrase "as shown in the following table" and the subsequent table all constitute document layout knowledge. For this type of knowledge, the system's priority is not to identify individual objects but to recover page structure and reading order.

The second category is chart numerical knowledge. It exists in financial reports, experimental reports, business dashboards, market analysis reports, and statistical charts. Its defining characteristic is that knowledge is not expressed directly in sentences but through visual encoding—for example, bar heights, line trends, color groupings, axis scales, and legend mappings. For this type of knowledge, the system must convert charts into structured data or chart semantic descriptions; otherwise, it is difficult to support precise question answering.

The third category is interface and object knowledge. It exists in software screenshots, operation manuals, industrial images, product photographs, and on-site photos. Its defining characteristic is that knowledge depends on object position, visual attributes, and spatial relationships. Expressions such as "upper-right button," "red warning icon," "submit button below the form," and "port on the left side of the device" can only be understood in conjunction with visual regions. For this type of knowledge, the system needs object detection, region description, and visual object alignment.

*Table 22-1: Major Forms of Visual Knowledge and Their Processing Priorities*

| Visual Knowledge Type      | Common Sources                              | Core Information                                         | Processing Priority                                  |
| -------------------------- | ------------------------------------------- | -------------------------------------------------------- | ---------------------------------------------------- |
| Document Layout Knowledge  | PDFs, scanned documents, contracts, reports | Heading hierarchy, paragraphs, tables, captions, page numbers | Layout parsing, reading order, region localization   |
| Chart Numerical Knowledge  | Financial reports, statistical reports, business dashboards | Axes, legends, trends, numerical relationships       | Chart recognition, data extraction, trend description |
| Interface & Object Knowledge | Software screenshots, operation manuals, product images | Buttons, icons, regions, object relationships       | Object detection, region description, spatial relationships |
| Image-Text Association Knowledge | Mixed image-text documents, manuals    | Body references, image captions, visual evidence         | Image-text alignment, captions, reference binding    |

This table illustrates that multimodal RAG is not a single task but a set of data engineering problems centered on visual knowledge organization. Different visual knowledge types require different processing strategies. If a system applies the same OCR-plus-text-chunking pipeline to all visual knowledge, its limitations will be continuously exposed in real-world scenarios.

---

### 22.1.4 Typical Failure Modes of Text RAG in Visual Scenarios

When text RAG is applied directly to multimodal documents or visual knowledge scenarios, the most common failure modes fall into four categories: missed detection, misreading, mislocalization, and evidence fragmentation.

Missed detection means the system entirely fails to capture critical information. For example, key metrics in a financial report are located in an image-based table; poor OCR quality or a parser that ignores image regions means this information never enters the knowledge base. Even if retrieval functions normally when the user asks a question, the system cannot recall knowledge that does not exist in the index.

Misreading means the system captured information but the structure or meaning is incorrect. For example, after a table is flattened into linear text, the system associates a value with the wrong field; axis scales in a chart are recognized incorrectly, causing a trend judgment to be wrong; the reading order of a multi-column page is scrambled, causing two unrelated paragraphs to be concatenated. Misreading is more dangerous than missed detection because the system generates plausible-sounding answers based on incorrect evidence.

Mislocalization means the system knows relevant content exists but cannot precisely locate it on the page or visual region. For example, the system responds "this data comes from Figure 3" but cannot identify which curve, which bar, or which table cell within Figure 3. This reduces the verifiability of the answer and makes it difficult for users to trust the system.

Evidence fragmentation means the link between text and visual evidence is lost. For example, the body text reads "As shown in Figure 4, model performance degrades under high load," but after parsing the body text chunk and the Figure 4 image chunk are not bound together. The system may recall the body text without recalling the image, or may recall OCR text from the image without associating the body text explanation. In this situation the model cannot make full use of the evidence.

The common thread among these failure modes is that the raw knowledge has not entirely disappeared, but it has not entered the retrieval system with the correct structure. They further confirm that the key challenge of multimodal RAG is not simply adding a visual model, but redesigning the data representation, indexing, and evidence organization of visual knowledge.

---

### 22.1.5 The Paradigm Shift from Text Chunks to Visual Chunks

The basic unit of text RAG is the text chunk, while the basic unit of multimodal RAG must be extended to the visual chunk. The development of visually-grounded language pretraining models demonstrates that treating page regions, visual patches, and text tokens collectively as modelable objects is an important technical foundation for moving from text chunks to visual chunks (Huang et al. 2022; Lee et al. 2023). A visual chunk is not simply a single image but a knowledge unit that carries a visual region, textual content, spatial position, semantic description, provenance information, and a citation anchor. For example, in a financial report PDF, a visual chunk can be an entire page, a table region, a chart, a footnote region, or a combination of a body paragraph and its corresponding chart. For a software operation manual, a visual chunk can be a screenshot region and its corresponding explanatory text. For a scanned contract, a visual chunk can be a signature region, an annotation region, or a table region.

A visual chunk should contain at minimum the following information: the image or page region itself, region coordinates, OCR text, visual description, associated text, the page it belongs to, the chapter path, content type, permission information, and a citation anchor. For chart-type chunks, it should also include chart type, axes, legend, extracted data, and a trend summary. For table-type chunks, it should include row-column structure, units, and remarks.

This means the knowledge unit schema of multimodal RAG will be more complex than that of text RAG. It must support not only text retrieval but also image retrieval, region localization, cross-modal reranking, and answer provenance. In the generation phase, the system must inform the model not only "what this passage says" but also "what this visual region represents, which texts it is related to, and where the user can verify it."

From this perspective, the fundamental change in multimodal RAG is the expansion of knowledge units from one-dimensional text fragments to two-dimensional or even multi-dimensional evidence objects. It requires data engineering to upgrade from "text chunking" to "page modeling" and "visual object modeling."

---

### 22.1.6 Section Summary

This section discussed why traditional text RAG cannot fully cover visual knowledge. The fundamental reason is that a large volume of real-world knowledge does not exist in linear text form but is embedded in page layouts, chart structures, screenshot regions, object positions, and image-text relationships. Although OCR can extract text from images, it cannot substitute for visual understanding because it cannot fully recover spatial structure, visual relationships, and cross-modal semantics.

The core challenge facing multimodal RAG is to organize visual knowledge into knowledge units that are retrievable, locatable, citable, and verifiable. This requires the system to move from text chunks to visual chunks, from pure-text indexes to cross-modal indexes, and from "finding relevant text" to "finding usable visual evidence."

The next section will further discuss visual chunk and object modeling, focusing on how page-level, region-level, object-level, and table-level chunks should be designed, and how bounding boxes, layout structures, captions, and OCR text can be organized into a unified multimodal knowledge representation.

---

## 22.2 Visual Chunks and Object Modeling

### 22.2.1 The Basic Concepts and Challenges of Visual Chunks

In multimodal RAG systems, understanding and processing visual information is a critically important engineering problem. Traditional text RAG systems build retrievable knowledge units by splitting documents into multiple text fragments (chunks) that typically rely on structured textual content such as paragraphs, tables, or sentences (Xu et al. 2020; Appalaraju et al. 2021). In visual scenarios, however, knowledge does not always appear in simple text fragments—especially in environments containing images, tables, charts, QR codes, complex layouts, or scanned documents.

Therefore, processing visual information goes beyond extracting the text in an image (i.e., OCR); it requires understanding the structure of the image itself, the spatial relationships among objects, and effectively aligning and fusing these visual elements with textual information. This gives rise to the concept of the visual chunk. A visual chunk refers to a basic visual unit extracted from an image; it may be a chart region, a photograph, a table, or any visually meaningful object. Analogous to traditional text chunks, visual chunks should contain the semantic information they carry, positional information, associated textual information, and more, to support subsequent retrieval, generation, and verification. The core characteristics of a visual chunk include the following types:

- **Diversity**: Visual chunks can take the form of images, charts, tables, flowcharts, etc., and contain multiple types of visual objects.

- **Spatial Relationships**: Visual chunks have spatial positional relationships, such as chart axes, object positions within images, and row-column arrangements in tables.

- **Text Association**: Many visual chunks contain content related to text, such as the headers and data in tables, and the labels and trends in charts.

- **Cross-modal Alignment**: Visual chunks must be aligned with their corresponding text to form a joint cross-modal representation, facilitating subsequent retrieval and generation tasks.

  ------



### 22.2.2 Visual Chunk Generation and Modeling Methods

In practical RAG systems, generating visual chunks involves several key steps, primarily including object detection and region segmentation, OCR extraction, and semantic annotation of visual regions (Ren et al. 2015; He et al. 2017; Kirillov et al. 2023). These steps work in concert to help the system identify the important visual elements in an image and create appropriate knowledge unit representations for them.

Object Detection is an important computer vision task in multimodal systems; it can identify different objects in an image and assign category labels to them. In RAG systems, object detection is typically used to identify visual regions in images such as charts, photographs, text boxes, and flowcharts. Region Segmentation further refines the work of object detection: it not only identifies objects but can also delineate their boundaries and determine the precise position of each object. Through region segmentation, the system can extract different regions from an image and align each region with its corresponding text or metadata.

OCR technology is mainly used to extract textual content from images, helping the system understand the text information within visual regions. In multimodal RAG systems, OCR does not merely extract raw characters from images but performs text extraction in conjunction with image structure, position, and context. For example, in an image of a financial statement table, OCR recognizes the data in the table and combines it with the table's row-column relationships to form structured tabular data. After OCR extraction, the system must also clean and normalize the textual content, including removing noise characters, correcting recognition errors, and fixing formatting issues. Furthermore, the OCR-extracted text must be aligned with other visual elements in the image to ensure correct matching between text and visual regions.

To make visual chunks more semantically meaningful, the system must also add appropriate labels and metadata to each visual region. These labels typically include the image category, the function of the region, and its relationship to associated text. For chart-type visual chunks, labels might include "axis," "trend line," and "legend"; for table-type visual chunks, labels might include "row header," "column header," and "value cell."

The introduction of these labels and metadata not only helps the system understand the content of each visual region but also facilitates subsequent retrieval and generation tasks. For example, when a user asks "How has the company's revenue changed over time?", the system can use the labels to identify the chart region associated with revenue and return it as evidence to the generation model, thereby generating an accurate answer.

---

### 22.2.3 Cross-modal Alignment: Joint Representation of Text and Vision

Cross-modal alignment is one of the core tasks in multimodal RAG systems; it requires the system to effectively combine and align visual information with textual information to provide accurate knowledge representations. Cross-modal alignment typically relies on a shared representation space for text and images. Works such as CLIP, ALIGN, and BLIP-2 have provided important foundations for text-vision joint representation and cross-modal retrieval (Radford et al. 2021; Jia et al. 2021; Li et al. 2023); its core objective is not merely to concatenate images and text, but to understand the relationships, interactions, and shared semantics between them.

One common alignment approach is position-based alignment. In this approach, the system uses position information of the OCR-extracted text to match text with regions in the image. For example, for a financial statement image, the system can use the row-column information of the table text to map OCR-extracted text to each cell region in the image. Another approach is semantics-based alignment. In this approach, the system considers not only the positional relationship between text and visual regions but also their semantic similarity. For example, in a report containing sales data and a trend chart, the system can identify the semantic association between the text "sales revenue" and the "sales revenue trend" in a line chart, thereby combining them into a complete visual chunk.

The joint representation of text and visual information is the ultimate goal of cross-modal alignment. Through joint representation, the system can process text and visual data simultaneously and provide integrated information at query time. For example, during retrieval, a user can simultaneously ask "What is the sales trend for Q1 2024?" and "In which chart is the sales data shown?" The system must, through joint representation, first identify the relevant sales trend chart in the image, then extract the description of the sales data from the text, and finally generate a complete answer. Joint representation typically employs multimodal embeddings, mapping text and visual information into the same vector space. In this space, the relationships between textual and visual elements are represented as distances or similarities between vectors, making cross-modal queries and retrieval more efficient.

![Figure 22-2: Joint Representation and Alignment of Text and Visual Elements](../../images/part7/图22_2zh.png)

*Figure 22-2: Joint Representation and Alignment of Text and Visual Elements*



---

### 22.2.4 Indexing and Retrieval Strategies for Visual Chunks

In multimodal RAG systems, the retrieval and indexing of visual chunks differ significantly from traditional text RAG. Visual chunks depend not only on the semantic information of text but must also account for image content, chart trends, spatial relationships, and more. Therefore, the indexing strategy for visual chunks must handle visual features, textual features, and metadata features, and support cross-modal retrieval (Radford et al. 2021).

Cross-modal indexing refers to indexing text and visual information together and creating a comprehensive index entry for each visual chunk. This entry contains not only textual content but also visual features (such as chart trends, table data cells, and image categories) as well as metadata (such as chart type, image dimensions, and chart title). In this way, the system can consider both text and visual information simultaneously during retrieval, resulting in better matching of user queries (Lee et al. 2023). A common approach to cross-modal indexing is to embed text and images separately into the same vector space and retrieve by computing the similarity between text and visual vectors. The advantage of this approach is its flexibility in handling information from different modalities and providing rich context for each query.

Multimodal retrieval refers to simultaneously using text and visual information during a query. For example, a user can input the query "What is the sales trend for Q1 2024?" along with a chart; the system must recognize the chart content and match it with the relevant sales data description to generate an accurate answer. To achieve this, the system must process both text queries and visual queries simultaneously and map them into the same retrieval space. The key to multimodal retrieval is designing an efficient cross-modal matching mechanism so that text and visual information can complement each other during retrieval. In practice, the system can first retrieve relevant content based on text search, then refine the results through image retrieval, and finally generate an answer by combining text and images.

---

### 22.2.5 Evaluation and Optimization of Visual Chunks

Evaluation and optimization are key steps for ensuring the long-term stable operation of a multimodal RAG system. In multimodal scenarios, evaluation must focus not only on the accuracy of retrieval results but also on the alignment between visual chunks and text, information completeness, and the credibility of final generated results—consistent with the emphasis that RAGAS and document VQA benchmarks place on evidence availability (Es et al. 2024; Mathew et al. 2021). Common metrics for evaluating multimodal RAG systems include:

- **Precision & Recall**: Evaluates system performance in visual and text retrieval.
- **Context Completeness**: Checks whether the system has provided a complete evidence chain and background information.
- **Generation Accuracy**: Evaluates whether the generated answer is based on correct evidence and is factually accurate.
- **Cross-modal Consistency**: Checks whether text and visual information are correctly aligned and whether key information is lost during generation.

Optimization of multimodal RAG systems must rely on evaluation feedback. The system should continuously optimize the visual chunk generation and modeling process based on evaluation results. For example, when the chart trends in a visual chunk are inconsistent with the text description, the system needs to adjust the chart recognition and trend analysis models; when the alignment between visual and text is low, the system needs to improve the joint representation method for text and vision. In this way, the system can gradually improve its cross-modal retrieval and generation capabilities as real-world usage scenarios evolve.

---

### 22.2.6 Section Summary

This section primarily discussed the generation and modeling of visual chunks in multimodal RAG systems. Through object detection, region segmentation, OCR extraction, and semantic annotation of visual regions, the system can transform images into semantically meaningful visual chunks, providing a reliable foundation for subsequent retrieval, generation, and traceability. The generation of visual chunks relies not only on recognizing image content but also on cross-modal alignment between images and text, thereby enabling precise multimodal retrieval and generation.

This section also explored cross-modal indexing and retrieval strategies in multimodal RAG systems, as well as how continuous evaluation and optimization can improve system stability and reliability. As multimodal capabilities are enhanced, RAG systems will be able to more comprehensively handle complex documents, reports, charts, images, tables, and other multimodal information, delivering higher-quality answers.

---

## 22.3 Cross-modal Indexing, Retrieval, and Reranking

### 22.3.1 Cross-modal Indexing and Joint Space

In multimodal RAG systems, the goal of cross-modal indexing is to manage and retrieve visual and textual data through an effective indexing mechanism. Unlike traditional text RAG systems that rely solely on textual information for indexing, cross-modal retrieval must handle the relationships between image and text data and store and index this information through a joint representation. To achieve this, systems typically embed images and text into a shared embedding space (Radford et al. 2021; Jia et al. 2021).

Specifically, the embedding space of images and text refers to a high-dimensional vector space into which the features of text and images are mapped. In this space, similar text and visual information will have nearby vector representations, enabling matching and retrieval by computing distances or similarities between vectors. To achieve cross-modal embedding of images and text, Dual-Stream Networks or Multimodal Embedding Models are typically employed. For example, text can be encoded by a pretrained language model (such as BERT or RoBERTa) to produce a vector representing text semantics, while images can be encoded by a visual model (such as ResNet or ViT) to produce a vector representing image features. The system then maps these two different-modality embedding vectors into the same shared space through joint training or alignment methods.

The advantage of the joint space is that it provides a unified framework for cross-modal retrieval. In this space, image and text information are represented as vectors and can be matched by computing their similarities. For example, a user can provide a text query "Q1 2024 sales growth," and the system can simultaneously retrieve both text passages and charts related to the query in the joint space, returning the most relevant images and text as results. Furthermore, the joint space can support cross-modal reranking. In multimodal queries, where a user may provide both images and text, the system computes similarities between text and images through the joint space and reranks the combined relevant images and text, thereby improving retrieval precision.

---

### 22.3.2 Visual Recall, Text Recall, and Cross-modal Reranking

In cross-modal retrieval, Visual Retrieval, Text Retrieval, and Cross-modal Reranking are three critically important stages. The system must first use visual recall and text recall to extract relevant content for the query, and then apply cross-modal reranking algorithms to refine the ranking of these results, thereby improving the relevance and accuracy of retrieval results.

#### 1. Visual Recall and Text Recall

Visual recall refers to image-based retrieval in response to a given query—finding the images most relevant to the query. Visual recall is typically accomplished by computing the similarity between an image's embedding vector and all image vectors in the image repository. Image embedding vectors can be extracted from images using visual models (such as ResNet or ViT) and then matched against the query's image vector to recall the most relevant images.

Text recall is similar to visual recall, except that it recalls text data. Given a text query, the system retrieves relevant text passages or documents by computing the similarity between the query text vector and the text vectors in the document repository. Text embedding vectors are typically encoded by a pretrained language model (such as BERT or GPT) and then matched against text vectors in the document repository.



*Table 22-2: Comparison of Visual Recall and Text Recall*

| Characteristic          | Visual Recall                                    | Text Recall                                         |
| ----------------------- | ------------------------------------------------ | --------------------------------------------------- |
| Primary Information Type | Image content, image features, visual patterns  | Semantic content, vocabulary, text passages         |
| Feature Extraction Method | Convolutional neural networks (CNN), vision transformers (ViT) | Word embeddings, sentence embeddings (BERT, RoBERTa) |
| Retrieval Method        | Similarity computation between image embedding and query image | Similarity computation between text embedding and query text |
| Cross-modal Fusion      | Based on visual information extraction and query matching | Based on semantic information extraction and text association |

#### 2. Cross-modal Reranking

Cross-modal reranking refers to performing a secondary ranking of recalled results on top of visual recall and text recall. This approach extends the paradigm of using neural models to rerank initial recall results from text retrieval (Nogueira and Cho 2019) into multimodal scenarios where visual signals are additionally incorporated. In multimodal queries, users typically provide both text queries and image queries; the system must determine the most relevant answers based on the results from both modalities.

Cross-modal reranking relies primarily on the image and text embedding vectors in the joint space. During the initial recall phase, the system first performs recall separately for text and images and treats the recalled results as a candidate set. During the cross-modal reranking phase, the system inputs the image and text embedding vectors together into a reranking model and sorts them according to their mutual similarities. Through this approach, the system can comprehensively account for the relevance of both images and text, improving the accuracy of retrieval results.

![Figure 22-3: Cross-modal Retrieval and Reranking Pipeline](../../images/part7/图22_3zh.png)

*Figure 22-3: Cross-modal Retrieval and Reranking Pipeline*



---

### 22.3.3 Retrieval Strategies for Complex Documents, Multi-page Reports, and Chart Q&A

When processing complex documents, multi-page reports, or Q&A tasks involving charts, traditional text RAG systems typically face challenges (Masry et al. 2022; Liu et al. 2023a; Liu et al. 2023b). Because these documents contain information from multiple modalities (e.g., text, tables, images, and charts), efficient retrieval and generation represent core challenges for multimodal RAG systems. To address this, systems require specialized retrieval strategies to support efficient multimodal information fusion.

#### Retrieval Strategies for Complex Documents

Complex documents (such as long documents, reports, and contracts) typically contain large amounts of information with complex hierarchical structures. When retrieving such documents in a multimodal RAG system, the system must effectively fuse different modalities of information including text, charts, images, and tables. To this end, the system should adopt a hierarchical retrieval strategy: first retrieve on textual content, then, based on the context of the retrieved text, search for relevant evidence in charts and images. In this way, the system can progressively narrow the retrieval scope and improve retrieval efficiency.

#### Retrieval Strategies for Multi-page Reports and Chart Q&A

Multi-page report and chart Q&A tasks typically require the system to simultaneously process content across multiple pages as well as data within charts. To address this problem, the system must implement cross-modal alignment between image regions and text, and retrieve by combining information from page-level, region-level, and chart-level chunks. During a query, the system can intelligently identify pages, charts, and table regions relevant to the query based on the user question's context, and quickly provide an answer.

For chart Q&A tasks, the system must pay particular attention to the relationship between charts and text. For example, in a financial report, a body text passage may describe data trends shown in a chart; the system must align the chart and text and, in combination with the data points in the chart, generate the final answer. This requires the system to simultaneously process images and text and effectively fuse the relevant information from both.

#### Joint Retrieval of Tabular and Chart Data

When processing complex documents containing both tables and charts, jointly retrieving tabular data and chart data becomes especially important. Traditional text retrieval systems typically ignore data points and trends in charts, while chart Q&A systems require specially designed cross-modal indexing and retrieval strategies to support joint retrieval of tabular and chart data. At query time, the system must combine numerical data and textual descriptions from both tables and charts to generate accurate answers.

---

### 22.3.4 Section Summary

This section explored key issues in cross-modal retrieval, including image embeddings, text embeddings, and the construction of a joint space. By embedding text and images into the same vector space, the system can simultaneously process multimodal queries and perform efficient retrieval. We also introduced the pipeline for visual recall, text recall, and cross-modal reranking, as well as how to improve the relevance and accuracy of retrieval results through joint space and reranking strategies.

In retrieval strategies for complex documents, multi-page reports, and chart Q&A, the system must combine multiple modalities of information—text, charts, tables, and images—and provide accurate answers through hierarchical and joint retrieval strategies. In the future, cross-modal retrieval technology will play an important role in processing multimodal scenarios such as complex documents, financial reports, and medical images.



---

## 22.4 Evaluation, Error Attribution, and Data Replenishment

### 22.4.1 Why Multimodal RAG Evaluation Cannot Focus Only on Whether the Answer Is Correct

In text RAG projects, many teams place the emphasis of evaluation on "whether the answer is correct"—for example, using human scoring, rule-based comparison, or an LLM-as-judge to determine whether the final response is correct. This approach, though incomplete, can more or less cover the main issues in pure-text scenarios. In multimodal RAG scenarios, however, focusing solely on the final answer readily masks structural defects within the system.

The reason for this phenomenon is that failures in multimodal RAG tend to occur in layers. A final incorrect answer may stem from a visual region not being recognized, from OCR misreading a character, from a visual chunk not entering the index, from a correctly recalled candidate being overridden by an interfering chart during cross-modal reranking, or from incorrect evidence assembly in the answer generation phase. If we only look at "right/wrong," we cannot determine whether the system stalled at parsing, indexing, retrieval, localization, or generation—and naturally we cannot optimize with precision. More importantly, multimodal systems frequently produce answers that "look approximately correct but cite the wrong evidence." For example, a user asks "What are the main reasons for the company's gross margin decline in Q2 2024?" The system's textual explanation is close to the correct answer, but the cited chart's page number is wrong, or it confuses the data regions of Q2 and Q3. If evaluation only measures semantic similarity, it may incorrectly judge this as correct; but in real business contexts, such a response cannot pass an audit and cannot build user trust.

Therefore, multimodal RAG evaluation must be upgraded from single-point answer assessment to chain-level evidence assessment. We typically decompose this into four layers: the first is the parsing layer, which focuses on whether pages, regions, tables, and charts have been correctly identified; the second is the retrieval layer, which focuses on whether relevant pages or regions have been recalled; the third is the localization layer, which focuses on whether the system can anchor the answer to the correct visual evidence; and the fourth is the generation layer, which focuses on whether the response is faithful to recalled evidence, complete, and verifiable. Only by observing all four layers simultaneously will system optimization have actionable leverage.

From an engineering perspective, this is equivalent to decomposing final answer quality into a multi-stage quality funnel. For each stage, separate metrics must be defined, separate datasets constructed, and the upstream-downstream dependencies recorded. In this way, the team can answer three key questions: First, at which stage is the problem most prominently exposed? Second, is this problem a universal defect or does it only appear in certain document types, chart types, or question formulations? Third, will fixing the current problem introduce new side effects?

To facilitate implementation, we can organize the core evaluation metrics for multimodal RAG into a joint scoring framework. Let the retrieval hit rate be $R_k$, the localization accuracy be $L$, the answer accuracy be $A$, and the evidence consistency be $E$; then the overall system score can be written as:

$$
S_{\text{mm-rag}}=\alpha R_k+\beta L+\gamma A+\delta E,\quad \alpha+\beta+\gamma+\delta=1
$$

The significance of this formula is not to give the system a single final score, but to remind us that an improvement in any high-level metric should not come at the cost of sacrificing lower-level verifiability. For example, certain generation optimizations may make answers appear superficially more fluent, but if evidence consistency decreases, the system's usability in high-risk scenarios is actually regressing.

It is important to note that $R_k$, $L$, $A$, and $E$ come from different evaluation layers and their raw values do not naturally share the same measurement scale, nor are they necessarily comparable across different datasets. For example, localization accuracy may depend on the granularity of bounding box annotations, evidence consistency may depend on human review standards, and answer accuracy may be influenced by the difficulty distribution of questions. Therefore, before using the weighted sum above, each metric must first be calibrated and normalized, with explicit statistical definitions, sample distributions, human calibration standards, and confidence intervals. Weights and metrics must be determined according to business risk, dataset distribution, and human calibration; cross-metric comparisons are invalid without normalization. This formula is better suited for version-to-version comparison and bottleneck analysis on the same evaluation set with the same statistical definitions, rather than being interpreted as an absolute score that can be directly compared across scenarios.

*Table 22-3: A Layered Perspective on Multimodal RAG Evaluation*

| Evaluation Layer | Focus                                 | Typical Issues                                        | Representative Metrics                                      |
| ---------------- | ------------------------------------- | ----------------------------------------------------- | ----------------------------------------------------------- |
| Parsing Layer    | Pages, regions, objects, tables, charts | Missed detection, incorrect detection, OCR misreading, structural fragmentation | Region recall rate, OCR character accuracy, table reconstruction rate |
| Retrieval Layer  | Page-level and region-level candidate sets | Relevant evidence not recalled, strong interference from negative samples | Hit@k, MRR, nDCG, cross-modal recall coverage rate          |
| Localization Layer | BBoxes, page numbers, chart cells    | Wrong page number, wrong region, wrong chart element localization | IoU, Grounding Accuracy, Page Localization                  |
| Generation Layer | Final response and cited evidence     | Hallucination, incorrect evidence citation, incomplete answer | Answer Accuracy, Faithfulness, Citation F1                  |

This table emphasizes the necessity of layered decomposition. For multimodal RAG, no single metric can replace a full system picture. Especially during enterprise deployment, if a team pursues only answer accuracy, it may achieve impressive results on demonstration data, but once the system encounters complex-layout PDFs, scanned reports, or mixed image-text documents, the fragility of the evidence chain will be rapidly exposed.

### 22.4.2 The Core Metric Framework: Hit Rate, Localization Accuracy, and Answer Accuracy

In practice, the three most commonly tracked metric categories are visual retrieval hit rate, localization accuracy, and answer accuracy (Ren et al. 2015; Mathew et al. 2021). They correspond respectively to the three questions "Was it found?", "Was it found precisely?", and "Was the answer correct?"—forming the backbone of a multimodal evaluation framework.

First is the visual retrieval hit rate. It focuses on whether, given a question, the system can recall the page, region, or chart containing the correct evidence into the top $k$ candidates. Unlike text RAG, the "correct evidence" in multimodal RAG is not necessarily a paragraph—it may be a chart region, a screenshot, or a set of table cells. Therefore, the annotation unit for Hit@k must be agreed upon in advance: whether it is page-level, region-level, or object-level. If the annotation granularity is inconsistent, the metric will be distorted.

A common definition is as follows:

$$
\text{Hit@}k=\frac{1}{N}\sum_{i=1}^{N}\mathbb{I}\left(\exists c \in C_i^{(k)},\ c \in G_i\right)
$$

where $C_i^{(k)}$ denotes the top $k$ candidates recalled for the $i$-th question and $G_i$ denotes the golden evidence set for that question. If at least one candidate among the top $k$ falls within the golden set, the question is counted as a hit. For multi-evidence questions, Evidence Recall can be further computed—i.e., what proportion of necessary evidence has been recalled.

Second is localization accuracy. It focuses on whether the system anchors the correct evidence to the correct region. For visual Q&A, this metric is critically important because many business applications are not satisfied with knowing "which page the answer is on" but require the system to point to "which column of the chart," "which cell of the table," or "which button in the screenshot." Localization accuracy is typically computed based on Intersection over Union (IoU):

$$
\text{IoU}(B_p,B_g)=\frac{|B_p \cap B_g|}{|B_p \cup B_g|}
$$

When the predicted bounding box $B_p$ and the golden bounding box $B_g$ have an IoU greater than the threshold $\tau$, the localization is considered successful. For page-level evidence, the threshold can be relatively lenient; for object-level or table cell-level evidence, a stricter threshold is typically required. Furthermore, in chart understanding scenarios, element grounding accuracy can be designed—for example, requiring the model to simultaneously identify the legend, axis, and target data series.

The third category of metrics is answer accuracy. It measures whether the final output satisfies business requirements. Surface-level semantic similarity alone is insufficient; the evaluation must distinguish factual correctness, evidence consistency, unit correctness, conditional correctness, and temporal correctness. For example, "revenue growth of 12%" and "net profit growth of 12%" may appear quite similar to a language model, but their business implications are entirely different. Likewise, "2023" versus "2024" differ by only one number yet can cause serious errors. Therefore, answer accuracy typically requires a combination of rule-based verification, human review, and LLM-as-judge methods.

In many projects, "answer accuracy" is further decomposed into two levels: one is strict accuracy, which requires that the facts, units, page numbers, and objects all be correct; the other is utility accuracy, which requires only that the main conclusions and business action recommendations be correct. The advantage of this decomposition is that the team can observe both the strict reliability of the system and the user-level usability, avoiding misguided optimization caused by an overly narrow metric definition.

*Table 22-4: Definitions and Applicable Scenarios for Key Evaluation Metrics*

| Metric Name            | Core Question   | Applicable Granularity  | Common Threshold or Determination Method     | Applicable Scenario                            |
| ---------------------- | --------------- | ----------------------- | -------------------------------------------- | ---------------------------------------------- |
| Hit@k                  | Was it found?   | Page-level, region-level | Whether the golden evidence exists in the top k candidates | Recall evaluation, index comparison            |
| Evidence Recall        | Is evidence complete? | Multi-evidence questions | Proportion of necessary evidence recalled    | Image-text joint Q&A, complex chart Q&A        |
| IoU / Grounding Acc.   | Was it found precisely? | Region-level, object-level | IoU > 0.5 / 0.7, etc.                       | BBox, table cell, button localization          |
| Answer Accuracy        | Was the answer correct? | Final response          | Rule-based verification + human/LLM judging  | End-to-end evaluation                          |
| Citation Faithfulness  | Was the evidence correct? | Response and citations  | Whether citations genuinely support the conclusion | Compliance Q&A, report Q&A, audit scenarios   |
| Time/Cost per Question | Is it worth deploying? | End-to-end pipeline    | Latency, token count, GPU cost               | Production environment optimization            |

In real systems, these metrics often do not change independently. For instance, increasing the recall candidate set from top 20 to top 100 may noticeably increase Hit@100, but the reranking burden grows, latency rises, and answer accuracy does not necessarily improve in tandem. Conversely, if we compress the candidate count to pursue low latency, complex chart Q&A recall may suffer. Therefore, evaluation metrics must not be interpreted in isolation but analyzed together with the pipeline structure.

![Figure 22-4: Multimodal RAG Evaluation Funnel](../../images/part7/图22_4zh.png)

*Figure 22-4: Multimodal RAG Evaluation Funnel*



### 22.4.3 Error Attribution for Missed Detection, Misreading, Mislocalization, and Chart Understanding Failures

With layered metrics in place, the next step is to establish an actionable error attribution framework. Error attribution is not simply recording "it was wrong" but stably mapping errors to fixable engineering causes. For multimodal RAG, the four most common first-level error types are: missed detection, misreading, mislocalization, and chart understanding failure.

Missed detection is the most fundamental and common problem. It means that the correct evidence never entered the candidate set. Missed detection can occur at multiple points: the page parsing phase fails to capture an entire page of images; the layout analysis phase fails to bound a table; image slicing at too coarse a granularity means fine-grained objects are not individually chunked; certain visual chunks are filtered out during indexing; query rewriting is too text-oriented, resulting in insufficient recall of visual candidates. The typical symptom of missed detection is: the user's question clearly corresponds to a chart or table in the document, but the system's returned candidates do not contain it at all.

Misreading means that evidence entered the system but its content was interpreted incorrectly. For example, OCR misreads "8.5%" as "3.5%"; table column headers are misaligned, causing "operating profit" to be associated with "operating revenue"; figure captions are incorrectly bound to adjacent images; content from the left and right columns of a multi-column layout is concatenated into a single passage. These issues are especially common in scanned documents, low-resolution images, and complex-layout documents. The danger of misreading is that the system will often produce self-consistent but incorrect answers—harder to detect than "I don't know."

Mislocalization is a high-frequency error unique to multimodal scenarios. The system may know that relevant information is on a certain page yet highlight the wrong region to the user; it may find the correct chart but use the wrong data series as evidence; or in interface screenshot scenarios it may mistake explanatory text above a button for the button itself. Mislocalization commonly occurs in systems where bounding box granularity is poorly designed, the layout model is unstable, or region-level rerank training is insufficient. It directly impacts explainability, because when users see evidence positions that are inconsistent with the answer, they immediately lose trust.

Chart understanding failure is a higher-level problem. It does not necessarily originate from OCR or localization errors but may stem from the model's inability to correctly extract trends, comparative relationships, outliers, or multi-series mappings. For example, the system can identify the legend entries "North America" and "APAC" and recognize the vertical axis scales, yet still selects the wrong series when answering "Which region had the fastest year-over-year growth?" This indicates that the failure point is not in text extraction but in chart semantic modeling.

To systematically convert these phenomena into engineering assets, we typically design an error coding table (Table 22-5) where each failure sample is tagged with at least three labels: first-level error type, specific sub-cause, and repair priority. First-level types are for macro-level statistics; sub-causes guide model or rule repairs; and priority helps the team schedule iteration order. For example, missed detection can be subdivided into "not sliced," "sliced but not indexed," and "indexed but not recalled"; misreading can be subdivided into "OCR character error," "header misalignment," and "caption binding error"; chart understanding failure can be subdivided into "axis misparse," "legend mapping error," and "trend description error."

*Table 22-5: Example Error Attribution Coding for Multimodal RAG*

| First-level Type        | Second-level Cause            | Typical Symptom                               | Priority Repair Direction                                        |
| ----------------------- | ----------------------------- | --------------------------------------------- | ---------------------------------------------------------------- |
| Missed detection        | Region not segmented          | Correct chart absent from candidate set       | Adjust layout detection, add region-level chunks                 |
| Missed detection        | Weak visual recall            | Page recalled, but chart region not recalled  | Strengthen visual embedding, add hard negatives                  |
| Misreading              | OCR error                     | Numbers, units, column headers recognized incorrectly | Replace OCR model, add post-processing correction rules    |
| Misreading              | Structure binding error       | Caption, footnote, image-text relationship mismatch | Layout relationship modeling, adjacency graph optimization  |
| Mislocalization         | Inappropriate bbox granularity | Highlighted region too large or overlaps adjacent region | Refine region hierarchy, train grounding model           |
| Chart understanding failure | Legend/axis/trend extraction error | Chart visible but correct comparisons/inferences unavailable | Add chart Q&A data, introduce structured extraction   |

When a team genuinely begins performing error attribution, they discover something very important: many problems that seem like "insufficient model intelligence" are actually caused by deficiencies in data structure design. For example, if a chart chunk does not separately store metadata such as legend, axis, and series, even the strongest model can only guess relationships from pixels and OCR text. In other words, error attribution is not just evaluation work—it is also a reverse check on whether data engineering has provided the model with sufficiently usable evidence structures.

![Figure 22-5: The Closed Loop from Error Attribution to Repair Actions](../../images/part7/图22_5zh.png)

*Figure 22-5: The Closed Loop from Error Attribution to Repair Actions*



### 22.4.4 From Online Failure Samples to Replenishment Datasets

The ultimate purpose of evaluation and attribution is not to produce an impressive analysis report but to drive continuous system improvement. The key improvement asset for multimodal RAG is typically not a one-time large-scale comprehensive benchmark set, but rather a replenishment dataset that is continuously accumulated from real online failures. Only online samples truly cover complex layouts, long-tail charts, low-resolution scans, enterprise-proprietary templates, and real user question formulations. Building hard negatives and replenishment datasets from these failure samples is an important practice for continuous improvement of retrieval systems and machine learning systems, and is also consistent with the fundamental requirements of data closed-loop governance in production machine learning (Sculley et al. 2015; Huyen 2022).

A mature failure sample replenishment pipeline typically consists of six steps. Step one: online log retention. The system records user questions, recalled candidates, cited evidence, final responses, user feedback, and human correction results. Step two: failure filtering. Through rules or human review, select samples that were answered incorrectly, cited incorrect evidence, mislocalized, or were "user-corrected after follow-up." Step three: error annotation. Apply labels according to the attribution framework defined earlier and supplement with golden evidence. Step four: sample reconstruction. Transform failure samples into data entries suitable for training and evaluation, including questions, documents, bounding boxes, chart structures, golden answers, and negative samples. Step five: data ingestion. Incorporate into replenishment dataset version management. Step six: targeted training and regression testing.

The most easily overlooked step in this pipeline is "sample reconstruction." Many teams feed raw online logs directly to the training pipeline and then find that performance improvement is limited. The reason is that raw logs typically contain only natural language questions and model outputs—they do not naturally constitute high-quality supervised data. For failure samples to truly be usable, their structural fields must be supplemented—for example, correct page numbers, correct visual regions, chart types, target cells, temporal conditions, units, business topics, and failure causes. Only in this way can a sample be used for both the retriever and for reranking, grounding, and generation alignment.

Another value of online failure samples is the ability to construct hard negatives. For visual retrieval, the hardest negative samples are usually not completely irrelevant images but regions that "look very similar but give the wrong answer." Examples include quarterly bar charts from the same financial report, screenshot regions from the same-template report in different sections, and similar tables on adjacent pages. Systematically incorporating these highly confusable negative samples into the training set tends to improve production results far more than blindly expanding general-purpose data.

In data engineering, the replenishment dataset is best managed with incremental versioning rather than repeated overwriting. Each sample should record at minimum: document source, question type, modality type, golden evidence, error label, repair status, and date first encountered. Only then can the team answer: Has a certain error type significantly decreased in a new version? Has a model upgrade fixed old problems while introducing new ones? Which business templates are under-covered in the current training set?

For replenishment priority, we typically consider not just frequency of occurrence but also business risk and difficulty of repair. A "slight page offset" occurring 10 times per month versus a "wrong unit on a financial metric" occurring 2 times per month—the latter is clearly higher risk. A priority function can be defined as:

$$
P_{\text{repair}} = \lambda_1 f + \lambda_2 r + \lambda_3 v - \lambda_4 c
$$

where $f$ represents failure frequency, $r$ represents business risk, $v$ represents the business value involved, and $c$ represents repair cost. This function helps the team allocate limited resources to problems that are "high-risk, high-value, and repairable," rather than being driven by noise samples.



### 22.4.5 Evaluation Set Design and Continuous Regression Mechanisms

For replenishment to produce stable returns, the evaluation set must evolve in parallel. A common misconception is that a team has only one static offline set on which the model repeatedly "improves its score," ultimately leading to poor generalization on complex online problems. Multimodal RAG is even less suited to this approach because page templates, chart styles, scan quality, and user question formulations are all continuously changing.

A more reasonable design approach is to adopt a layered evaluation set. The first layer is the foundational capability set, covering low-level capabilities such as OCR, table parsing, chart extraction, and bounding box alignment. The second layer is the pipeline set, covering page recall, region recall, cross-modal reranking, and evidence assembly. The third layer is the business set, covering real-world scenarios such as financial reports, policy documents, product manuals, and screenshot Q&A. The foundational set is used for diagnosing model components; the pipeline set is used for verifying data engineering improvements; and the business set is used for judging whether the system is truly usable.

For regression mechanisms, every model or index version upgrade should run at least three types of regression: first, a full benchmark regression to confirm there is no significant degradation; second, an error cluster regression specifically observing whether previously high-frequency failure patterns have improved; and third, a near-online new sample regression verifying that the system can cover the latest document templates and question formulations. If only the overall average score is tracked, it is easy to encounter the illusion of "overall improvement of 1 point, but key business scenarios regressed by 8 points."

Additionally, multimodal scenarios should pay particular attention to consistency between chunk and evidence versions. Suppose a PDF has been re-parsed once and its page slicing method has changed; the bounding boxes, chunk IDs, and page-level coordinates in the old evaluation set may then be invalid. If the evaluation foundation is not synchronized with the index version, "pseudo-errors" will appear where "the system actually answered correctly, but the annotation baseline has expired." Therefore, the version numbers of evaluation data must be managed together with the parser version, index version, and chart extraction version.

In engineering practice, a useful approach is to store "question–document–evidence" in a decoupled manner: the question set is maintained separately, document parsing results are versioned independently, and evidence annotations are referenced through stable anchors. This structure reduces the risk of an entire evaluation set becoming obsolete due to minor layout changes and is better suited for long-term operations.

### 22.4.6 Section Summary

This section covered evaluation, error attribution, and data replenishment for multimodal RAG. Unlike text RAG, multimodal systems cannot focus solely on whether the final answer is correct; they must simultaneously observe four layers: parsing, retrieval, localization, and generation. Visual retrieval hit rate, localization accuracy, and answer accuracy form the most basic metric backbone, while missed detection, misreading, mislocalization, and chart understanding failure constitute the most common error attribution framework.

More importantly, the endpoint of evaluation is not to produce a score but to form a sustainable data replenishment closed loop. Only by stably converting online failure samples into high-quality training and regression assets can the system progressively adapt to the long-tail complexity of real enterprise documents. The next section will further summarize, from a case study and patterns perspective, reusable designs for multimodal RAG in scenarios such as financial report assistants and complex-layout enterprise document retrieval.

---

## 22.5 Case Studies and Pattern Summary

### 22.5.1 Why Case Summaries Are More Valuable Than Point Solutions

Reaching the final section of Chapter 22, we have already discussed the problem definition of multimodal RAG, visual chunk design, cross-modal retrieval, and evaluation and data replenishment. The next question to answer is more deployment-oriented: when these capabilities truly enter a business scenario, how should they be combined? Which parts must be rebuilt? Which parts can be reused? Which patterns have been validated in practice and are most worth crystallizing as templates?

This is the value of the "case studies and pattern summary" section. For engineering teams, isolated technical conclusions are often insufficient to support complete deployment. For instance, we know that page-level and region-level chunks are needed, and we know that visual recall and reranking are needed, but this does not directly translate into a deployable system. What truly determines success or failure is usually how these technical components are combined into a pipeline oriented around business objectives, and how tradeoffs are made among cost, latency, accuracy, and explainability.

From experience, multimodal RAG has at least three common deployment paths. The first is the "page-primary, region-supplementary" document Q&A path, suitable for complex PDFs, policy documents, and scanned materials. The second is the "chart and table centric" analytical Q&A path, suitable for financial reports, business analysis, and experimental reports. The third is the "screenshot and object centric" operational assistance path, suitable for product manuals, software guides, and interface Q&A. The three paths share the same underlying multimodal capabilities but differ notably in slicing granularity, index organization, evaluation standards, and interaction patterns.

Therefore, the purpose of a case summary is not simply to showcase "what a project accomplished" but to abstract projects into transferable patterns. A good pattern summary should answer four questions: What is the true knowledge unit in the scenario? Where is retrieval most likely to fail? What special requirements does the generation phase place on evidence organization? If this approach is transferred to another business, which components can be directly reused and which must be re-annotated and retrained?

### 22.5.2 The Financial Report Assistant Case

The financial report assistant is one of the most typical scenarios for multimodal RAG—and one that most readily exposes system limits. The challenge is not an extremely large volume of documents, but rather that knowledge is scattered, evidence is heterogeneous, and user questions typically carry strong analytical intent. An "explanatory" question often requires the system to simultaneously leverage body text analysis, financial tables, quarterly charts, note disclosures, and cross-page metric comparisons (Masry et al. 2022; Liu et al. 2023b).

Taking the user question "What are the main reasons for the company's gross margin decline in Q2 2024?" as an example, an ideal system must complete at least five tasks. First, identify the retrieval anchors: "gross margin," "Q2," and "causes of decline." Second, locate the management discussion and analysis passage in the body text. Third, find the corresponding quarterly data in a table or chart. Fourth, confirm that these evidence items are consistent in temporal terms and definitional scope. Fifth, assemble quantitative evidence and qualitative explanation into a citable answer. If any one step goes wrong, the answer may be superficially plausible but fundamentally unreliable.

The most effective modeling approach for a financial report assistant is typically not to treat an entire annual report as a "long document" but to decompose it into three categories of collaborative evidence: body text explanatory evidence, numerical table evidence, and chart trend evidence. Body text evidence answers "why"; table evidence answers "how much"; chart evidence answers "whether the trend holds." At the retrieval pipeline level, the system can first use text retrieval to identify the main theme passage, then use the metric names, time periods, and chart numbers appearing in those passages to trigger region-level table/chart recall in reverse.

A key pattern in such systems is "text-guided vision"—rather than relying purely on visual embeddings to solve all problems. The reason is that chart styles in financial reports are highly templated; relying solely on image similarity easily confuses adjacent quarters, similar metrics, or same-template charts. By extracting anchors such as "gross margin," "Q2," "cost pressure," and "product mix changes" from the body text and then using these anchors to narrow the visual search space, accuracy can typically be significantly improved.

Another lesson is that the financial report assistant must explicitly handle units, definitional scope, and time dimensions. For example, "year-over-year growth," "quarter-over-quarter growth," "percentage point change," "billion yuan," and "million yuan" can all be used interchangeably. If the system retrieves solely based on text similarity, it easily assembles evidence with inconsistent definitional scopes. Therefore, visual chunks in financial report scenarios are best annotated with structured metadata such as reporting period, currency, unit, metric alias, chart number, and page range.

*Table 22-6: Evidence Organization Patterns in the Financial Report Assistant*

| Evidence Type           | Primary Question Answered | Typical Source                     | Retrieval Strategy                              | Role in Generation                              |
| ----------------------- | ------------------------- | ---------------------------------- | ----------------------------------------------- | ----------------------------------------------- |
| Body text explanatory evidence | Why                | Management discussion, risk disclosures | Text retrieval + topic expansion             | Generate qualitative explanation                |
| Table numerical evidence | How much               | Income statement, segment table, notes | Structured table retrieval + region localization | Provide numerical values, units, YoY/QoQ basis |
| Chart trend evidence    | Whether the trend holds   | Quarterly charts, business dashboards | Chart region recall + cross-modal reranking   | Support trend inference and visual citation     |

In the answer generation phase, financial scenarios should not allow the model to "freely summarize"; a more reliable approach is to use evidence template-based assembly. For example: first output the conclusion statement, then the data statement, and finally the evidence source statement. This has two advantages: first, it reduces the probability of the model mixing different quarters or definitional scopes; second, it facilitates displaying page numbers, table titles, or chart numbers in the response, improving audit traceability.

### 22.5.3 The Complex-Layout Enterprise Document Retrieval Case

Complex-layout enterprise document retrieval relies on capabilities such as heading hierarchy, table boundary detection, footnote binding, and region localization—exactly what models such as the LayoutLM family and DocFormer focus on (Xu et al. 2020; Appalaraju et al. 2021; Huang et al. 2022). Unlike financial reports, the core difficulty of complex-layout enterprise document retrieval is not necessarily chart understanding, but rather that page structure is extremely complex, document templates are non-standardized, and many questions depend on "relationships within a page" rather than a single text fragment. Typical documents include procurement/tendering files, scanned contracts, policy manuals, product compliance materials, technical specifications, and mixed image-text SOPs.

The most common challenge in this scenario is: the user asks "whether a certain condition is satisfied" or "where a certain requirement appears," and the answer is often scattered across tables, footnotes, appendices, and body text. For example, "bid bond amount" in a tendering document may appear in one place in the body text, one place in a table, and one place in a footnote; "liability for breach of contract" in a contract may be interspersed across multiple pages of clauses with annotations or signature page references. If the system still cuts documents into ordinary text passages, evidence fragmentation will occur frequently.

Therefore, the most important aspect for this type of document retrieval is not a more powerful generation model, but more robust page modeling. The system needs to restore heading hierarchy, table boundaries, image-text adjacency relationships, header/footer filtering rules, and cross-page continuation relationships. Often, the most valuable engineering improvement is not swapping in a stronger visual model but adding layout relationship fields to chunks, such as "continues from/to previous/next page," "table heading attribution," "footnote binding region," and "same-clause cross-page anchor."

Such systems also require special emphasis on "localization verifiability." Enterprise users often want to know not just the answer but "on which page, in which clause, in which table cell did you derive this conclusion." Therefore, in complex-layout retrieval scenarios, page number, clause number, table title, region screenshot, and highlighted localization are often just as important as the answer itself. Without localization capability, multimodal retrieval is merely a fancier Q&A system rather than a knowledge tool usable for business decision-making.

In practice, one effective pattern is the "two-part answer": first give the conclusion, then give the evidence localization. For example: "This policy requires outsourced vendors to pass an annual review. Evidence is located on page 14, section 4.2, in the 'Review Period' row of the subsequent table." This answer structure is particularly user-friendly because it both reduces the cognitive burden and preserves a verification entry point.

### 22.5.4 Mapping to the P05 Project

Within the project framework of this book, the methodology of Chapter 22 does not exist in isolation—it directly provides the data and pipeline design foundation for subsequent project work. If mapped to a project analogous to the P05 multimodal knowledge assistant in Part 14 of this book, it can be roughly decomposed into six deliverable modules: document ingestion and parsing, visual chunk construction, cross-modal indexing, retrieval orchestration, evidence generation and citation, and evaluation-replenishment closed loop. Readers can first understand the data objects, indexing, and evaluation methods of multimodal RAG in this chapter, and then consult the corresponding project chapter for the P05 project's end-to-end structure, implementation steps, and deliverable organization.

The significance of this mapping is that project teams do not need to start from scratch to understand what "multimodal" means; they can directly translate the methods of Chapter 22 into engineering to-dos. For example, Section 22.2 corresponds to data layer schema design; Section 22.3 corresponds to online retrieval orchestration; Section 22.4 corresponds to the evaluation and operations closed loop; and Section 22.5 abstracts all these modules into project templates. This significantly reduces the situation of "understanding all the technology but never being able to assemble the project."

If we further refine down to deliverables, we can write the key build items of P05 as pattern cards. Each card contains: target scenario, input document types, primary knowledge unit, recommended slicing granularity, index key, retrieval order, core metrics, and replenishment mechanism. The value of pattern cards is that different business teams can directly reuse and customize them as needed, rather than reinventing the approach each time.

A common engineering formula is to evaluate whether a particular pipeline design is worth deploying. Let the answer accuracy gain be $\Delta A$, the evidence verifiability gain be $\Delta E$, the latency cost be $T$, and the per-query cost be $C$; a deployment utility function can then be defined as:

$$
U_{\text{deploy}} = \eta_1 \Delta A + \eta_2 \Delta E - \eta_3 T - \eta_4 C
$$

This function reminds us that multimodal solutions should not pile up all models in pursuit of "technical sophistication," but should instead seek the optimal balance among accuracy, explainability, and cost. For example, for a system with only a small number of screenshot Q&A tasks, it may not be necessary to introduce a complex chart extraction pipeline; whereas for a financial assistant, the business returns from evidence verifiability are sufficient to justify higher inference costs.

*Table 22-7: Mapping from Chapter 22 Methods to P05 Project Modules*

| Chapter 22 Method Module          | P05 Corresponding Build Item        | Key Deliverables                                         | Primary Acceptance Metrics                            |
| --------------------------------- | ----------------------------------- | -------------------------------------------------------- | ----------------------------------------------------- |
| Visual chunk & object modeling    | Multimodal data schema              | Page blocks, region blocks, chart blocks, table blocks   | Parse completeness rate, structural consistency       |
| Cross-modal indexing & retrieval  | Retrieval service & rerank service  | Vector index, keyword index, candidate merger            | Hit@k, MRR, region-level recall rate                 |
| Evidence localization & answer generation | Citation & highlight component | Page numbers, bboxes, screenshots, highlight anchors    | Citation Faithfulness, IoU                            |
| Evaluation & error attribution    | Quality operations backend          | Evaluation set, error labels, failure sample repository  | Answer Accuracy, error cluster decline trend          |
| Data replenishment                | Training data pipeline              | Hard negatives, replenishment sample versions            | Version iteration gains, online repair coverage rate  |

### 22.5.5 Reusable Design Patterns and Anti-patterns

After repeated validation across multiple projects, we can summarize the experience of multimodal RAG into several stable patterns. The first is "coarse page-level recall + fine region-level localization." This applies to nearly all complex document scenarios because it balances recall efficiency with evidence explainability. The second is "text-guided vision," which means using text topic anchors to narrow the visual retrieval scope before performing region-level comparison—particularly suited for financial reports and manuals. The third is "metadata-first," which means explicitly storing page numbers, chapter numbers, chart identifiers, units, time ranges, and object categories at indexing time, rather than placing all hope on the generation model to understand these on the fly.

There is also a very important pattern called "evidence before answer." This means the system pipeline is organized first around evidence, and only afterward around linguistic expression and polish. For high-risk businesses, whether the response is eloquent is not the top priority; whether the evidence is authentic, verifiable, and highlightable to the user is what truly determines whether the system can be deployed.

Corresponding to these patterns, there are also several high-frequency anti-patterns. The most typical anti-pattern is "treating OCR as multimodal." This causes the system to continuously fail on charts, interface screenshots, and complex-layout scenarios. The second anti-pattern is "indexing only at page level." While whole-page indexing is simple to implement, once a page has high information density, the model during generation has great difficulty focusing precisely on the target region. The third anti-pattern is "offline evaluation only, no online replenishment." Such systems often perform well during prototype validation but quickly suffer continuous failures in enterprise real-world templates due to accumulating long-tail errors.

*Table 22-8: Reusable Patterns and Common Anti-patterns for Multimodal RAG*

| Type         | Name                                          | Effect or Consequence                               | Explanation                                                     |
| ------------ | --------------------------------------------- | --------------------------------------------------- | --------------------------------------------------------------- |
| Pattern      | Coarse page-level recall + fine region-level localization | Balances efficiency and explainability | Find the page first, then find the bounding box—suitable for complex documents |
| Pattern      | Text-guided vision                            | Improves accuracy in similar-chart and similar-screenshot scenarios | Use text anchors to narrow visual candidate range        |
| Pattern      | Metadata-first                                | Reduces unit, temporal, and definitional confusion  | Preserve page numbers, chart numbers, units, etc. at index time |
| Anti-pattern | Treating OCR as multimodal                    | Frequent failures on charts and layout problems     | Can read characters but cannot understand visual structure      |
| Anti-pattern | Indexing only at page level                   | Evidence granularity too coarse, generation easily confused | Particularly prone to mislocalization on high-density pages |
| Anti-pattern | No online replenishment                       | Long-tail problems do not converge                  | Offline score high but production quality unstable              |

From an organizational collaboration perspective, reusable patterns also have a practical value: they help product, algorithm, data, and engineering teams build a shared vocabulary. Once all stakeholders understand what "coarse page-level recall + fine region-level localization" means and what "evidence before answer" means, many project debates will shift from abstract concepts to verifiable designs, significantly improving execution efficiency.



### 22.5.6 Section Summary

This section reviewed the deployment methodology of multimodal RAG from the perspective of case studies and patterns. The financial report assistant emphasizes the collaboration of three evidence types: textual explanations, table numerical data, and chart trends. Complex-layout enterprise document retrieval emphasizes page structure recovery, clause localization, and verifiable answers. The mapping to P05 further translates the methodology of Chapter 22 into project deliverable modules.

More worth crystallizing are the cross-project reusable patterns: coarse page-level recall and fine region-level localization, text-guided vision, metadata-first, and evidence before answer. Against these patterns stand the anti-patterns: treating OCR as multimodal, indexing only at page level, and lacking online replenishment. By mastering these patterns, teams can not only build a demonstrable multimodal Q&A system but also progressively construct a multimodal RAG platform that is verifiable, operable, and continuously optimizable in real business contexts.

------

## Chapter Summary

This chapter established that when knowledge is hidden in images, tables, charts, and layout structures, text RAG—which flattens documents into character strings—discards visual structure and image-text alignment information; therefore, the key to multimodal RAG lies in reconstructing knowledge units, indexing strategies, and evidence organization. Toward this end, the chapter presented a design for visual chunks and object modeling that organizes pages, regions, tables, and images into locatable, citable, and verifiable knowledge assets, and explained how cross-modal indexing, retrieval, and reranking bring visual evidence into the recall-and-generation loop.

At the evaluation level, the chapter decomposed visual retrieval errors into parsing, localization, recall, ranking, and generation stages for attribution, and used online failure sample replenishment to continuously improve system capability in complex document scenarios. These methods collectively form the data engineering foundation for multimodal financial report assistants and complex enterprise document retrieval applications. After deployment, real users and a continuously evolving knowledge environment will keep generating new failure signals; the next chapter transitions to the online feedback loop and knowledge updates.

## References

Xu Y, Li M, Cui L, Huang S, Wei F, Zhou M (2020) LayoutLM: Pre-training of Text and Layout for Document Image Understanding. In: Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pp 1192–1200.

Huang Y, Lv T, Cui L, Lu Y, Wei F (2022) LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking. In: Proceedings of the 30th ACM International Conference on Multimedia, pp 4083–4091.

Appalaraju S, Jasani B, Kota B U, Xie Y, Manmatha R (2021) DocFormer: End-to-End Transformer for Document Understanding. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp 993–1003.

Kim G, Hong T, Yim M, Nam J, Park J, Yim J, Hwang W, Yun S, Han D, Park S (2022) OCR-free Document Understanding Transformer. In: Proceedings of the European Conference on Computer Vision (ECCV), pp 498–517.

Mathew M, Karatzas D, Jawahar C V (2021) DocVQA: A Dataset for VQA on Document Images. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pp 2200–2209.

Mathew M, Bagal V, Tito R, Karatzas D, Valveny E, Jawahar C V (2022) InfographicVQA. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pp 1697–1706.

Radford A, Kim J W, Hallacy C, Ramesh A, Goh G, Agarwal S, Sastry G, Askell A, Mishkin P, Clark J, Krueger G, Sutskever I (2021) Learning Transferable Visual Models From Natural Language Supervision. In: Proceedings of the 38th International Conference on Machine Learning (ICML), pp 8748–8763.

Jia C, Yang Y, Xia Y, Chen Y-T, Parekh Z, Pham H, Le Q, Sung Y-H, Li Z, Duerig T (2021) Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision. In: Proceedings of the 38th International Conference on Machine Learning (ICML), pp 4904–4916.

Li J, Li D, Savarese S, Hoi S (2023) BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models. In: Proceedings of the 40th International Conference on Machine Learning (ICML), pp 19730–19742.

Lee K, Joshi M, Turc I, Hu H, Liu F, Eisenschlos J, Khandelwal U, Shaw P, Chang M-W, Toutanova K (2023) Pix2Struct: Screenshot Parsing as Pretraining for Visual Language Understanding. In: Proceedings of the 40th International Conference on Machine Learning (ICML), pp 18893–18912.

Masry A, Long D X, Tan J Q, Joty S, Hoque E (2022) ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning. In: Findings of the Association for Computational Linguistics: ACL 2022, pp 2263–2279.

Liu F, Piccinno F, Krichene S, Pang C, Lee K, Joshi M, Altun Y, Collier N, Eisenschlos J M (2023a) MatCha: Enhancing Visual Language Pretraining with Math Reasoning and Chart Derendering. In: Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics, pp 12756–12770.

Liu F, Eisenschlos J M, Piccinno F, Krichene S, Pang C, Lee K, Joshi M, Chen W, Collier N, Altun Y (2023b) DePlot: One-shot Visual Language Reasoning by Plot-to-Table Translation. In: Findings of the Association for Computational Linguistics: ACL 2023, pp 10381–10399.

Ren S, He K, Girshick R, Sun J (2015) Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. In: Advances in Neural Information Processing Systems 28, pp 91–99.

He K, Gkioxari G, Dollár P, Girshick R (2017) Mask R-CNN. In: Proceedings of the IEEE International Conference on Computer Vision (ICCV), pp 2961–2969.

Kirillov A, Mintun E, Ravi N, Mao H, Rolland C, Gustafson L, Xiao T, Whitehead S, Berg A C, Lo W-Y, Dollár P, Girshick R (2023) Segment Anything. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp 4015–4026.

Nogueira R, Cho K (2019) Passage Re-ranking with BERT. arXiv preprint arXiv:1901.04085.

Es S, James J, Espinosa-Anke L, Schockaert S (2024) RAGAS: Automated Evaluation of Retrieval Augmented Generation. In: Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics: System Demonstrations, pp 150–158.

Sculley D, Holt G, Golovin D, Davydov E, Phillips T, Ebner D, Chaudhary V, Young M, Crespo J-F, Dennison D (2015) Hidden Technical Debt in Machine Learning Systems. In: Advances in Neural Information Processing Systems 28, pp 2503–2511.

Huyen C (2022) Designing Machine Learning Systems: An Iterative Process for Production-Ready Applications. O'Reilly Media.
