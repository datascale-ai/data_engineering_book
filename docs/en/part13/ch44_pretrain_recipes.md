# Chapter 44: LLM Pre-training Data Engineering in Practice: From Recipe to Implementation

## Opening Scenario: The Night a 1B-Token Reproduction Failed

At two in the morning, the loss curve on the training dashboard still had not reached the expected range. Several engineers stood around the meeting-room display. On the left were logs from a completed 1B-token run; on the right were the Llama-3 technical report and several public reproduction notes. The original plan had sounded simple: run a small proportional reproduction. If large models can be trained on trillions of tokens, then shrinking data scale to 1B tokens and scaling model size, batch size, learning rate, and data ratios should produce a roughly similar trend. At minimum, the loss curve should decline smoothly and validation performance should not plateau so early.

Reality did not follow that reasoning. The first 20% of tokens looked normal: the model quickly learned common lexical patterns and short sentence structure. In the middle phase, validation loss began to fluctuate, perplexity on code data dropped slowly, and math samples showed almost no benefit. In the later phase, the model continued memorizing frequent expressions, but long-context ability, reasoning questions, and instruction samples did not improve in step. The most painful result appeared at evaluation time: another small model trained on the same 1B-token budget but with a staged data recipe was more stable on several downstream tasks.

At first, the team blamed hyperparameters. They changed warmup, adjusted peak learning rate, and tried smaller batches. Some suspected tokenizer mismatch, some suspected over-cleaning, and others argued that the code-data share was too low. After several follow-up experiments, the issue became clearer: the failure was not a single parameter. The team had understood "recipe" too coarsely. A data recipe is not just mixing web, code, math, encyclopedia, books, and QA data according to one global ratio and feeding that mixture uniformly throughout training. The ratios reported by large-model teams are closer to a final cross-section than to a construction blueprint.

Small-scale training amplifies this problem. Large-scale training has enough token budget for a model to first absorb broad language distributions and then gradually encounter high-quality knowledge, code, math, and instruction data. A 1B-token budget is short. If all data types are mixed from the first step, the model is pushed into difficult and stylistically diverse samples before it has formed a stable language foundation. Low-quality web data consumes early learning windows, while high-value samples appear too sparsely to form sustained gradient signal. The nominal ratios are correct, but the training rhythm is wrong.

That night, the team redrew the data flow. The first phase no longer chased full coverage; it emphasized clean general text to establish basic language ability. The second phase gradually increased encyclopedia, books, and code weights, exposing the model to denser knowledge and structured expression. The third phase introduced math, complex reasoning, and high-quality instruction samples, using a smaller but cleaner set to drive targeted ability. Deduplication, quality scoring, language identification, toxicity filtering, and difficulty stratification stopped being one-time preprocessing steps and became sampling decisions inside every phase.

The lesson was direct: a training recipe cannot be treated as a static ratio table. Ratios answer what data exists and roughly how much each category contributes. They do not answer when to feed the data, at what quality threshold, at what difficulty, or with what repetition and downweighting strategy. What truly affects training is a staged data pipeline: raw corpora enter a candidate pool, then pass through filtering, deduplication and clustering, domain classification, difficulty assessment, and phase-specific sampling. For teams trying to reproduce models such as Llama-3, reducing the token budget does not mean scaling every setting down proportionally. Smaller reproduction runs need clearer training rhythm because every token affects ability formation earlier and more directly.

Unlike blind trial-and-error that collapses under simple scaling, DeepSeek-V3 (Liu et al. 2024) and Qwen2.5 (Hui et al. 2024) show strong awareness of data staging. They do not train from beginning to end on one fixed mega-mixture. Early pre-training uses large amounts of broad-domain foundational knowledge and code. High-quality synthetic long text and math data are introduced more precisely in the middle and late stages. This intuition, dynamically matching data mixture to the model's developmental stage, reduces distribution-shock risk and helps reasoning ability scale more steadily.

## 44.1 The Public Data Transparency Spectrum

Before discussing concrete pre-training recipes, we need an information-quality yardstick. As competition intensifies, "open-source models" differ widely in what they disclose about training data. Without filtering the language of technical reports, engineers can easily mistake public-relations phrasing for actionable data engineering guidance. We classify data disclosure along four dimensions: **sources**, **mixture / ratio**, **cleaning pipeline**, and **downloadability**.

![Figure 44-1: Data Recipe Funnel](../../images/part11/29_1_data_recipe_funnel_en.png)

*Figure 44-1: The data recipe funnel.*

The funnel in Figure 44-1 narrows sharply. Macro numbers such as 14.8T tokens are only the surface layer. Inferring precise domain ratios goes deeper. The details that truly map to engineering actions are often obscure heuristic thresholds and cleaning scripts.

1. **White-box transparency.** These models publish not only papers but the entire data factory from crawling to final packaging. Sources can be traced to specific Common Crawl dumps, mixture ratios are precise, cleaning scripts are open, and direct Hugging Face downloads are provided. OLMo (Groeneveld et al. 2024) and some early Amber models are typical examples. These are the most valuable data assets for the open community.
2. **Grey-box partial transparency.** Most leading open-weight models, including DeepSeek-V3, Qwen2.5, and Llama-3, fall here. Their reports list macro data categories such as web, code, and math, and may disclose relative ratios or total scale. They do not release the actual cleaning pipeline code, nor do they provide the final cleaned high-quality corpus. Reproduction means reading the recipe and sourcing the ingredients yourself.
3. **Black-box opacity.** Reports only vaguely mention a large, high-quality multilingual corpus. There is no source breakdown, no concrete ratio, and no code or data download. The disclosed numbers are often useful for publicity but weak as engineering guidance.

![Figure 44-2: Large-Model Data Transparency Spectrum](../../images/part11/29_2_data_transparency_spectrum_en.png)

*Figure 44-2: Large-model data transparency spectrum.*

> **Note:** Chapters 4, 5, and 6 already covered the general foundations of data acquisition, cleaning, MinHash LSH deduplication, and tokenization. The source map from Chapter 4 is the base layer for this chapter. Here we focus on recipe-level engineering tradeoffs among model families.

![Figure 44-3: Pre-training data source map](../../images/part11/4_1_pretrain_data_source_map.png)

*Figure 44-3: Pre-training data source map.*

**Table 44-1: Data transparency spectrum for mainstream open LLMs**

| Model family | Source-category disclosure | Mixture-ratio disclosure | Cleaning rule / code disclosure | Pre-training data downloadable | Overall transparency |
| --- | --- | --- | --- | --- | --- |
| DeepSeek-V3 | Detailed macro categories | Macro ratios disclosed | High-level strategy only | No | Grey-box, partially transparent |
| Qwen2.5 family | Detailed macro categories | Evolution and ratios disclosed | Classifier and deduplication strategy described | No | Grey-box, partially transparent |
| Qwen3 (expected) | Detailed macro categories | Evolution and ratios disclosed | New cleaning pipeline described | No | Grey-box, partially transparent |
| Llama-3.1 / 3.3 | Public macro categories | Phase token ratios disclosed | Staged cleaning and schedule described | No | Grey-box, partially transparent |
| OLMo-2 | Public macro categories | Ratios by stage disclosed | Code and strategy open | Partially | Open, high transparency |

## 44.2 Cross-Model Data Composition

After peeling away report packaging, the core question is what each team actually put into the furnace. Table 44-2 summarizes pre-training data composition based on public reports, model-behavior inference, and community reproduction practice. It extends the data-source, license, and risk matrix introduced in Chapter 4.

Numbers use the following tags: **[D]** = directly disclosed in a report; **[I]** = reasonable inference from public model behavior and context; **[E]** = community or author estimate.

**Table 44-2: Pre-training data composition of mainstream open LLMs**

| Data category | Subtype / feature | Quality requirement | DeepSeek-V3 (14.8T) | Qwen2.5 (18T) | Qwen3 (inferred) | Llama-3.1 / 3.3 (inferred) | OLMo-2 (inferred) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **General web** | High-quality web pages | Very high; deduplicate and remove content farms | ~50% [I] | ~55% [I] | ~50% [E] | ~50% [I], staged use, heavier early | High share, strict deduplication and multi-level filtering |
| **Chinese web specialization** | News, forums, encyclopedias | Semantically complete, spam removed | High share, specialized cleaning [D] | Very high share, multi-level deduplication [D] | Stricter filtering [E] | High share, multi-stage cleaning, stronger deduplication [D] | High share, multi-level deduplication, special spam filtering [D] |
| **Books** | E-books and publications | Copyright compliance, long logic | ~5-10% [E] | ~5% [E] | ~5% [E] | ~5-10% [E], high-quality late-stage data | ~5-8% [E], used in Dolmino Mix quality stage |
| **Code** | Source, issues, PRs | Multilingual, repository structure preserved | ~15-20% [E] | ~18% [I] | ~20% [E] | ~15% [E], staged enhancement for structure learning | ~15-20% [E], repo structure and language coverage preserved |
| **Math and logic** | Forums, synthetic derivations | Clean formatting, rigorous logic | ~10-15% [E] | ~12% [I] | ~15% [E] | ~10% [E], high-quality math concentrated late | ~10-15% [E], used in Dolmino quality stage |
| **Academic** | arXiv, medical, legal | Terminology, formula parsing | ~5% [I] | ~5-8% [I] | ~8% [I] | ~5% [I], high-quality journals and public papers | ~5-8% [I], concentrated in academic stage of Dolmino Mix |

![Figure 44-4: Estimated data mixture ratios](../../images/part11/29_3_models_pie_chart_en.png)

*Figure 44-4: Estimated mixture-ratio comparison across model families.*

The comparison suggests three counterintuitive findings.

1. **Expansion of code and math data is a foundation for intelligence emergence.** In early LLMs such as GPT-3 or LLaMA-1, code and math accounted for only about 5% of the corpus and were treated as vertical capability supplements. In DeepSeek-V3 and Qwen2.5, their combined share approaches or exceeds 20-30% [E]. Engineers found that dense logical and structured code data is fuel for general reasoning, not only for code generation. The strict syntax tree structure of code implicitly forces causal structure learning.
2. **The effective share of general web text is falling generation by generation.** Total token counts have risen to 14.8T and beyond, but the relative share of casual web chatter and unstructured documents is declining. Teams would rather spend compute deduplicating and filtering away 80% of web data than consume training windows that could be used by high-quality academic and synthetic data. Scaling laws are shifting from pure scale to effective information-density scale.
3. **Synthetic data is quietly taking over the middle and late stages of pre-training.** Natural high-quality human data, especially math steps and reasoning chains, is close to exhaustion. DeepSeek and Qwen both introduce large amounts of synthetic QA and math derivations generated by earlier strong models in the middle and late stages. These samples fill the logical gaps left by natural corpora.

## 44.3 Chinese-English Ratio and Multilingual Strategy

Open models now split into two strategic camps: Chinese-English dual-core models and English-dominant models. Qwen2.5 and DeepSeek-V3 are the most successful and useful representatives of the first camp.

### Qwen2.5's Broad Multilingual Strategy

Qwen2.5 was not designed as a simple bilingual model. It aimed to cover more than 29 major languages. Data engineering therefore faced a hard problem: high-quality data for long-tail languages is scarce, and low-quality multilingual mixing can cause negative transfer, sometimes called an alignment tax.

Qwen addressed this with complex language sampling weights. For ultra-high-resource languages such as English, aggressive MinHash LSH deduplication (Broder 1997) and downsampling reduce homogeneous overfitting to the same world knowledge. For low-resource but high-quality languages, such as high-quality Arabic encyclopedia data, document-level upsampling increases exposure. Qwen also uses multilingual alignment corpora, including parallel machine-translation data and multilingual Wikipedia, and improves token compression for many languages. This lowers inference cost for low-resource languages and improves cross-lingual knowledge transfer.

### DeepSeek-V3's Chinese-English Dual-Core Strategy

DeepSeek-V3 chooses a sharper strategy. It preserves strong Chinese understanding and alignment, while the foundation of reasoning and logic is built from large, high-quality English community data, especially code and academic documents.

In the recipe, DeepSeek does not chase absolute Chinese-token dominance. The team understands that high-quality science and engineering discussions, open-source code, and academic papers are still overwhelmingly in English. Chinese data is therefore cleaned more aggressively: high-threshold quality classifiers remove enormous amounts of marketing content and content farms. The remaining Chinese corpus is smaller but much purer, enough to anchor Chinese values and expression habits. Reasoning depth is driven by high-quality English tokens. This "English for the brain, Chinese for the mouth" recipe is a classic way to balance resource efficiency and model intelligence.

## 44.4 The Transfer Dividend of Code Data

The low-level collection sources, deduplication logic, and general code corpora such as The Stack were covered in Chapter 4. Here we focus on higher-level engineering tradeoffs.

In top-tier model development, specialized coder models and base models are no longer separate technology trees. Training a dedicated coder model and feeding its lessons back into the base model's logical ability is the transfer dividend of code data.

### From DeepSeek-Coder to V3

DeepSeek had already validated the value of using DeepSeek-Coder as a pathfinder during the V2 era. In V3, code data was no longer a raw dump of GitHub. The team built fine-grained dependency graphs through repository-level parsing.

If code is fed as loose file-level text chunks, the model learns fragmented syntax. Only when cross-file concatenation follows repository dependency order, defining structures before interfaces call them, can the model learn project-level architectural logic. This topological ordering at the file level is one reason DeepSeek-V3 retains strong code capability at low compute cost.

### Qwen2.5-Coder Co-Evolution

Qwen uses the same idea. Qwen2.5's base corpus includes several trillion tokens of filtered code data. It cleans GitHub source code and also structures technical blogs, StackOverflow QA, and Jupyter Notebook interaction records containing code snippets. This text-code interleaved data is an excellent bridge between fuzzy human needs and precise machine instructions.

### Llama-3 Code Data Strategy

Llama-3, including 3.1 and 3.3, introduces large-scale code data during long-context training. Sources include public repositories and high-quality code collections such as GitHub projects, public issue trackers, and pull request logs. The goal is to provide structured, parseable logic patterns so the model can learn function definitions, variable scope, nested logic, and common algorithms.

Code data is staged. Early training uses general web text and long documents to build language foundations. Mid-training gradually introduces code and math for reasoning and structured understanding. Late annealing resamples and upsamples high-quality code to strengthen complex logic and multi-step computation. Staging prevents code from disrupting natural language learning too early while preserving performance in both short- and long-context tasks.

Packing also matters. Llama-3 isolates code packing by preserving continuity within a project or file and avoiding arbitrary concatenation across projects or syntax environments. This reduces semantic conflict and context jumps. RoPE is extended during code stages so the model can capture function-call relations, loops, and class inheritance over longer contexts.

### OLMo-2 Code Data Strategy

OLMo-2 is more transparent. Its code data includes multilingual source code, issue tracking, code review logs, open-source library documentation, and programming QA from StackExchange. It uses a two-stage curriculum: broad web data first, then Dolmino Mix high-quality data where code is concentrated for capability compression. Code enters the second stage in structured and deduplicated form, preserving functions and logic blocks.

OLMo-2 emphasizes multilingual coverage and project structure preservation. Samples keep directory and file hierarchy where possible so the model can learn modular design, dependencies, and naming conventions. Strict deduplication prevents template memorization, while complete logic and comments are preserved for mathematical derivations and algorithmic implementations.

**Table 44-3: Code data sources and scale for mainstream models**

| Model family | Estimated total code scale | GitHub source share | Interactive notebook / QA share | Repo-level parsing | Formatting strategy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **DeepSeek-V3** | ~2.5T tokens [E] | ~70% [E] | ~30% [E] | Strong support, topological sorting [D] | Preserve indentation and directory tree |
| **Qwen2.5** | ~3T tokens [E] | ~65% [E] | ~35% [E] | Strong support, combined with FIM [I] | Preserve Markdown and notebooks |
| **Llama-3.1 / 3.3** | ~2.8T tokens [I] | ~70% [I] | ~30% [I] | Strong support, project and file integrity preserved [I] | Preserve indentation, function blocks, and directory tree [I] |
| **OLMo-2** | ~2T tokens [E] | ~60% [E] | ~40% [E] | Strong support, project-level parsing [D] | Preserve indentation, directory structure, and notebook content [D] |

## 44.5 Synthetic Evolution of Math and Reasoning Data

Injecting strong mathematical genes during pre-training is central to reducing hallucination and logical disorder. Because naturally occurring, well-formatted math solution processes are scarce, synthesis and cleaning become the main tools.

### DeepSeekMath's Verifier Pipeline

DeepSeek's rapid math improvement depends heavily on the DeepSeekMath corpus (Shao et al. 2024). Generic crawlers often corrupt MathJax and LaTeX into unreadable fragments. DeepSeek built specialized DOM parsers for HTML math tags and, more importantly, introduced rule-based verification through reinforcement learning and formal proof tools.

The pipeline recalls candidate math text from massive internet corpora, then uses smaller scoring models to filter for "gold" samples with complete derivation logic. Prior-generation models repeatedly solve open problem sets; only solution traces that pass absolute verification in a SymPy sandbox or Lean prover are fed back into pre-training. This greatly improves sensitivity to long-chain mathematical proof.

### Qwen2-Math's Synthetic Strategy

Qwen relies even more on synthetic data feedback. Public Qwen2.5-Math materials show extensive use of Qwen-Max or earlier strong models as teachers to generate multi-step CoT solutions for base math questions.

To avoid distribution collapse from synthetic data, the team uses answer-consistency checks. If several derivations for the same problem converge to the same unique and correct solution in a sandbox, the synthetic reasoning path may enter pre-training or post-training data. Math-data scale growth is therefore no longer limited by the speed of human-written internet content; it is directly driven by compute.

## 44.6 Long-Text Data Source Analysis

Long-text data is a core resource for context understanding, long-range dependency learning, and complex reasoning. This section analyzes three strategies: natural long documents, synthetic long text, and short-document concatenation.

### Natural Long Text

Natural long text usually comes from e-books, academic papers, technical manuals, public reports, and high-quality web articles. Its advantage is structural completeness, semantic coherence, and clear logic. Novels, technical documents, and academic papers often contain thousands or tens of thousands of tokens in sequence. During pre-training, these texts help the model learn topic development, paragraph logic, and reasoning chains, improving long-summary, cross-paragraph QA, and complex instruction following.

The limitation is scarcity, especially for high-quality and clearly licensed data. Styles also vary greatly across papers, news, and manuals, which can introduce bias toward particular writing forms. Training therefore needs staged sampling, text normalization, and deduplication to maintain diversity and semantic consistency.

### Synthetic Long Text

Synthetic long text has become an effective way to expand long-context training. Existing models or rule systems generate long continuous documents when natural long data is scarce or copyright-limited. Researchers can control topic, length, and structure to satisfy token and context-span needs at different training stages. Quality scoring, content filtering, and diverse generation can improve usefulness, but generated long text still carries risks: weak logic, semantic repetition, and unnatural context transitions. It must be filtered with automatic checks and human sampling.

### Short-Document Concatenation

Short-document concatenation combines multiple short fragments into long sequences. Sources can include news paragraphs, forum posts, QA pairs, code comments, and social posts. Packing should consider document boundaries, topic consistency, and length control so the sequence satisfies context-length needs without destroying coherence.

The advantage is flexibility and broad coverage. Concatenation quickly expands trainable long-text quantity and introduces multiple sources and styles. During pre-training, it can combine with packing and staged scheduling to isolate document boundaries or mix topics for efficient long-context learning. The risk is that concatenation cannot fully match natural long-document coherence. Excessive packing can create context jumps, logical breaks, and topic drift. Quality control, topic clustering, transition generation, and duplicate filtering help reduce this noise.

### Combined View

In practice, the three strategies are mixed. Natural long text provides coherence and logic. Synthetic text fills the gap left by scarce high-quality long data. Short-document packing increases volume and style diversity. With staged scheduling, quality filtering, and context-length control, training data can cover long-context summarization, cross-paragraph reasoning, instruction following, and complex QA.

This multi-strategy approach reflects a principle of large-model data engineering: balance quantity with quality, and structure with coverage.

**Table 44-4: Long-context data strategies of mainstream models**

| Model family | Maximum window | Long-text sources | Short-document packing | RoPE scaling and tuning stage | Performance penalty control |
| --- | ---: | --- | --- | --- | --- |
| DeepSeek-V3 | 128K [D] | Long books / repo-level code | Cross-document packing isolation | Late pre-training annealing extension [D] | YaRN (Peng et al. 2023), very low accuracy loss |
| Qwen2.5 | 128K [D] | Long reports / synthetic long text | Strict EOD-token isolation | Progressive window extension [I] | YARN / dynamic base-frequency adjustment |
| Llama-3.1 / Llama-3 Herd | 128K [D] | Multilingual web / code / math / long-context continued pre-training data | 8K sequences in base stage; gradually longer sequences in long-context stage; document boundaries should be preserved [I] | RoPE base frequency raised to 500K; six stages from 8K to 128K; about 800B tokens of long-context continued pre-training [D] | Check short-context recovery and Needle-in-a-Haystack pass rate at each window; final 40M-token annealing, high-quality upsampling, and checkpoint averaging [D] |
| OLMo-2 | 4K [D] | DCLM web / StarCoder code / academic papers / arXiv STEM / Wikipedia and Wikibooks / StackExchange / synthetic math | Fixed 4096-token sequences; packing details not fully expanded in the report, can be implemented with document boundaries plus EOS isolation [I] | RoPE with theta raised to 500K; no public 128K extension stage; focus on Dolmino Mix 1124 mid-training and annealing [D] | RMSNorm, QK-Norm, and Z-loss improve stability; repeated n-grams filtered; Dolmino high-quality mid-training and checkpoint soup reduce regression [D] |

Very long context is not introduced at full scale from the beginning. In the final stage of DeepSeek-V3 training, the team supports a 128K context window through dynamic long-text extension. It selects structurally complete long books, technical manuals, and merged repo-level code, and modifies RoPE base frequency (Su et al. 2024). By gradually extending sequence length from 4K in the final training stage, the model transfers local short-range dependency logic to very long contexts while saving substantial compute.

## 44.7 Curriculum Multi-Stage Training Schedule

Curriculum schedule is not a decorative trick. It is how a data recipe becomes executable. Static ratios only describe what is in the training set. The schedule decides when each corpus appears, at what weight, with what context length, and under what learning-rate strategy. The most common reproduction mistake is treating public ratios as an all-run uniform sampling table. Real large-model training usually stages ability formation: large general corpora build language foundations; high-quality data, long-context data, code, and math add targeted capability; annealing and post-training stabilize the model.

### Llama-3: Long-Context Extension and Final Annealing

The most useful Llama-3 case to inspect is Llama-3.1. Base pre-training is still next-token prediction. The model first learns broad language, knowledge, code, and math distributions under an 8K context window. The goal is not immediate ultra-long context; it is stable short-context ability. Attention cost grows quadratically with length, so introducing 64K or 128K sequences too early would be expensive and could inject ineffective long-document noise before language structure is stable.

Llama-3.1 therefore enters long-context continued training late. Public reports describe six stages that extend from the original 8K context to 128K. Each step trains until the model adapts before moving to the next length. Adaptation is measured by short-context evaluation recovery and stable performance on long-context retrieval tasks such as needle-in-a-haystack. Context length itself is a curriculum difficulty dimension.

Another important element is annealing. The public report states that the final 40M tokens linearly anneal the learning rate to zero while keeping the 128K window, upsampling high-quality sources, and averaging checkpoints from the annealing process. This is a final shaping stage: earlier broad corpora provide coverage, continued long-context training provides window ability, and annealing stabilizes the final checkpoint.

### OLMo-2: A Reproducible Two-Stage Curriculum

OLMo-2 is valuable because it discloses more of the training process and works well as a reproduction reference. It uses a two-stage design. Stage one uses a large web-heavy mixture: about 4T tokens for 1B and 7B models, and about 5T tokens for 13B. This stage builds language foundations, common knowledge, and broad coverage.

Stage two switches to smaller but higher-quality data, Dolmino Mix 1124. The OLMo-2 repository gives more detail: the 1B model uses about 50B high-quality tokens multiple times; the 7B model also uses about 50B high-quality tokens with different data orders and model soups; the 13B model uses several 100B-token high-quality branches plus one 300B-token branch, then averages weights. The point is not merely more training. Different data orders and high-quality late-stage data reduce randomness, while checkpoint or weight averaging improves robustness.

OLMo-2 offers a transparent template: use large-scale data for coverage, then high-quality target data for capability compression. It does not scatter all high-quality data evenly across the whole run. Concentrating it late makes gradient signal denser once the model already has basic language ability. This is especially important for smaller reproductions with limited token budgets.

### Qwen2.5: Continuous Schedule Across Pre-training, Long Context, and Post-training

Qwen2.5's schedule has three layers. The first is base pre-training scale expansion. The report states that high-quality pre-training data grew from 7T tokens in the previous generation to 18T tokens, improving common sense, professional knowledge, and reasoning. "High quality" means not just clean text, but domain organization, quality filtering, and ratio control. Qwen2.5 must support general QA, code, math, multilingual ability, and structured data understanding, so the recipe cannot be web-text centered.

The second layer is long-context training. Public Qwen2.5 models support long context, and Qwen2.5-1M further shows how long-context ability can be extended with synthetic long data, progressive pre-training, and multi-stage SFT. This resembles Llama-3.1: do not push context length to the limit at the start. Let the model adapt to shorter lengths first, then expose longer sequences. Synthetic long data fills task-structure gaps in long QA, summarization, cross-paragraph retrieval, and codebase reasoning.

The third layer is post-training. Qwen2.5's post-training includes more than one million complex SFT samples and multi-stage reinforcement learning. It improves not only chat preference, but long-text generation, structured-data analysis, and instruction following. Post-training should not be treated as a simple alignment add-on. It continues the curriculum: SFT supplies controllable task formats, while preference learning or RL adjusts style, reliability, and complex-task performance. If long-context ability formed during pre-training, post-training must include long-context tasks or the model may drift back toward short-instruction behavior.

### Implications for Reproduction

Llama-3, OLMo-2, and Qwen2.5 point to the same conclusion: an effective data recipe is a staged pipeline, not a static table. The first stage solves coverage. The second solves quality and domain ability. The long-context stage solves input length and global dependency. Annealing or high-quality late-stage training stabilizes the model. SFT and RL solve usability and instruction behavior.

For 1B-token or smaller reproductions, do not scale large-model data ratios down mechanically. Split the token budget into phases: clean general text early, gradually increase code, math, encyclopedia, and books in the middle, and concentrate high-quality data late for annealing or capability compression. If long-context ability is required, schedule a separate window-extension phase and monitor both short-context evaluation and long-document retrieval. The purpose of schedule is to make the model meet the right data difficulty at the right time.

![Figure 44-5: Llama-3 annealing data timeline](../../images/part11/29_4_llama3_annealing_schedule_en.png)

*Figure 44-5: Llama-3 annealing-period data composition timeline.*

Qwen2.5 also reflects curriculum learning (Bengio et al. 2009). In the foundation phase, massive web and base corpora teach statistical language distribution and common sense. In the high-quality refinement phase, quality thresholds rise sharply, general text declines, and code, math, and rigorous academic documents increase. In the annealing and ultra-long-context phase, learning rate decreases and more synthetic data, high-quality human instruction data, and very long sequences bridge pre-training into alignment.

### 44.8 Cases, Pitfalls, and Boundaries

Theory and implementation often diverge. Three cases illustrate the gap.

**Case A: reverse-engineering DeepSeek.** A team tried to reproduce an internal model by inferring the data recipe and training strategy from public papers and logs. Initially, it mixed long books, repo-level code, and QA data directly according to paper ratios. Under a small token budget, performance fell far below expectation. The missing element was staging. Early training needed general text for stable language foundations; mid-training should introduce structured code and math; late training should raise the share of high-quality instruction data. Without staging, gradient signal was diluted and the model failed to learn higher-order logic and long-context dependency.

**Case B: Qwen vocabulary expansion to 152K.** The team tried to increase vocabulary coverage for multilingual and professional-domain understanding, but short-sequence performance dropped. The issue came from mismatch between short-document packing and the RoPE scaling stage. With short token lengths, enlarged RoPE parameters made the model over-sensitive to position encoding, and some semantic information was flattened. Poor cross-document packing isolation also caused hallucination in cross-document contexts. Vocabulary and context-window expansion must be coordinated with schedule and short-document processing.

**Case C: OLMo-2 as an open reference.** OLMo-2 uses a two-stage curriculum: broad web and multilingual data first, then Dolmino Mix high-quality data to compress ability. The comparison shows that the main difference between open data and closed internal data lies in quality control and staging. Open data has explicit layers and token-distribution monitoring for each stage. Closed environments need additional dynamic sampling and weighting for multiple tasks and long context.

Common pitfalls include missing staging, poor short-document packing, premature RoPE or position-extension work, uneven high-quality data distribution, sequence length conflicting with batch configuration, over-cleaning that reduces generalization, and lack of checkpoint averaging or model soup. For small reproductions at or below 1B tokens, staged data flow is mandatory. Long-context tasks should use progressive RoPE and long-text training to avoid short-sequence regression. Multi-source mixing must coordinate with packing and weights, and high-quality samples should be concentrated late rather than spread thinly from the beginning.

Further reading includes the Llama 3 Herd report for staged training, long-context extension, and annealing checkpoint averaging; OLMo-2 logs for Dolmino Mix and the two-stage curriculum; and Qwen2.5 papers and model cards for high-quality multi-source sampling, vocabulary expansion, and long-context schedule. A single data-ratio table is never enough. Stable reproduction under limited token budgets requires schedule, packing, context-window extension, and high-quality sample timing to be designed together.

## References

Bavarian M, Jun H, Tezak N, Schulman J, McLeavey C, Tworek J, Chen M (2022) Efficient Training of Language Models to Fill in the Middle (FIM). arXiv preprint arXiv:2207.14255.

Bengio Y, Louradour J, Collobert R, Weston J (2009) Curriculum Learning. In: Proceedings of the 26th Annual International Conference on Machine Learning, pp 41-48.

Broder A Z (1997) On the Resemblance and Containment of Documents. In: Proceedings of the Compression and Complexity of Sequences, pp 21-29.

Dubey A, Jauhri A, Pandey A, Kadian A, Al-Dahle A, Letman A, Mathur A, Schelten A, Yang A, Fan A, others (2024) The Llama 3 Herd of Models. arXiv preprint arXiv:2407.21783.

Groeneveld D, Magnusson I, Bhagia A, Schwenk D, Soldaini L, Tafjord O, Sherborne M, Kinney R, Authur C, Atkinson D, others (2024) OLMo: Accelerating the Science of Language Models. In: Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics, pp 15789-15809.

Hoffmann J, Borgeaud S, Mensch A, Buchatskaya E, Cai T, Rutherford E, de Las Casas D, Hendricks L A, Welbl J, Clark A, others (2022) Training Compute-Optimal Large Language Models (Chinchilla). arXiv preprint arXiv:2203.15556.

Hui B, Yang J, Cui Z, Yang J, Liu D, Zhang L, Liu B, Yu B, Lu K, Chi K, others (2024) Qwen2.5: A Party of Foundation Models. arXiv preprint arXiv:2412.15115.

Kaplan J, McCandlish S, Henighan T, Brown T B, Chess B, Child R, Gray S, Radford A, Wu J, Amodei D (2020) Scaling Laws for Neural Language Models. arXiv preprint arXiv:2001.08361.

Liu A, Feng B, Xue B, Wang B, Wu B, Lu C, Zhao C, Deng C, Zhang C, Ruan C, others (2024) DeepSeek-V3 Technical Report. arXiv preprint arXiv:2412.19437.

Peng B, Quesnelle J, Fan H, Shippole E (2023) YaRN: Efficient Context Window Extension of Large Language Models. arXiv preprint arXiv:2309.00071.

Sennrich R, Haddow B, Birch A (2016) Neural Machine Translation of Rare Words with Subword Units (BPE). In: Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics, pp 1715-1725.

Shao Z, Wang P, Zhu Q, Xu R, Song J, Zhang M, Li Y, Wu Y, Guo D (2024) DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv preprint arXiv:2402.03300.

Su J, Lu Y, Pan S, Murtadha A, Wen B, Liu Y (2024) RoFormer: Enhanced Transformer with Rotary Position Embedding (RoPE). Neurocomputing 568:127063.

Wang X, Wei J, Schuurmans D, Le Q, Chi E, Narang S, Chowdhery A, Zhou D (2023) Self-Consistency Improves Chain of Thought Reasoning in Language Models. In: International Conference on Learning Representations.
