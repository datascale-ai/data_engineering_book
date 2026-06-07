# Chapter 29: Data Valuation and Reuse Mechanisms

## Chapter Guide

Chapters 27 and 28 explained how to make data discoverable, understandable, and dependable through catalogs, metadata governance, data products, and data contracts. Once data can be found and trusted, a sharper question appears: how much is this data actually worth? What benefits does it create, what costs does it consume, and which assets deserve continued investment rather than merely looking large?

This question cannot be answered by intuition. Many organizations still judge data value through convenient but fragile proxies: more data must be more valuable, expensive purchased data must be important, or a dataset that raises one model metric must be worth keeping. These assumptions often misallocate resources. Teams repeatedly clean and store low-value data while neglecting the corpus that supports many downstream scenarios. They buy data for a one-time metric improvement, then leave it unused and unmaintained, turning it into technical debt (Sculley et al. 2015).

In large-model data engineering, acquisition, labeling, compliance, and maintenance costs rise quickly, while the marginal contribution of data is highly nonlinear. Data valuation therefore becomes a core engineering problem, not a finance exercise. This chapter builds a measurable and comparable valuation method, then shows how deliberate reuse can amplify one asset across pre-training, post-training, RAG, evaluation, and compliance.

The chapter proceeds in four movements. First, it analyzes why data value is often misjudged. Second, it defines a multidimensional metric system covering reuse rate, coverage, training gain, retrieval hits, labeling savings, risk reduction, and maintenance cost. Third, it explains the five major reuse paths for a large-model data asset. Finally, it works through an enterprise-domain corpus case, then turns the method into portfolio tools: a cost-benefit matrix, an asset review card, an engineering pipeline, and governance mechanisms.

## 29.1 How Data Value Is Misjudged

### 29.1.1 Four Common Value Illusions

Before defining good valuation, it is useful to name the bad shortcuts. The idea that information should be measured and managed as an asset is not new (Moody and Walsh 1999; Laney 2017), but in practice valuation often collapses into easy proxies. These proxies are dangerous because they look correlated with value in ordinary situations, which makes them feel like substitutes for valuation.

*Table 29-1: Four common illusions about data value*

| Illusion | Naive assumption | Why it fails |
| --- | --- | --- |
| Scale illusion | More data is more valuable | Marginal returns decline; redundancy and low quality can reduce performance |
| Cost illusion | Expensive data is valuable data | Past spending is sunk cost, not downstream benefit |
| Model-gain illusion | Any metric lift means value | Local, short-term, replaceable gains are not asset value |
| Business-value illusion | Model metrics translate directly into business results | There is often a wide gap between offline metrics and business outcomes |

These illusions frequently reinforce one another. Their common root is replacing difficult-to-observe real value with easy-to-observe proxy quantities.

### 29.1.2 The Scale Illusion: More Data Is Not Automatically More Value

The scale illusion has a plausible source. During pre-training, model performance often improves smoothly as data, parameters, and compute grow, a relationship captured by neural scaling laws (Kaplan et al. 2020). This strengthened the belief that more data is always better.

Scaling laws, however, describe statistical relationships under broadly comparable data quality and distribution. They do not guarantee that arbitrary additional data creates value. Compute-optimal training results show that model size and data volume must grow together under a fixed compute budget; blindly increasing one dimension has rapidly diminishing returns (Hoffmann et al. 2022). Deduplication research shows that reducing duplicated data can improve model quality while reducing data volume (Lee et al. 2022). Data pruning research further shows that selecting high-information samples can beat naive power-law scaling with less but better data (Sorscher et al. 2022). In an even sharper direction, carefully selected small high-quality corpora can match much larger ordinary corpora for some training goals (Gunasekar et al. 2023).

The conclusion is direct: value is not proportional to size. It depends on information density, quality, coverage, and relevance. A small, deduplicated, balanced, accurately labeled corpus can be more valuable than a huge corpus filled with repetition, noise, and distribution skew. Data-centric AI emphasizes systematic quality improvement for exactly this reason (Zha et al. 2023).

### 29.1.3 The Cost Illusion: Input Is Not Output

The cost illusion treats acquisition cost as value. It appears most often in external data purchases and large labeling projects: a costly dataset feels "important" and receives protection that may exceed its actual contribution.

Cost and value are different quantities. Acquisition cost answers how much the organization paid to obtain the data. Value answers how much downstream benefit the data creates. A costly industry-report corpus may be nearly worthless if it does not match business scenarios and is never reused. Conversely, an internal interaction log may cost little to collect but become a critical asset because it matches the real production distribution.

The danger is the sunk-cost fallacy. Because the organization already spent money, it keeps paying for cleaning, storage, compliance review, and maintenance even when the asset has little forward-looking value. Acquisition cost should be used to review past procurement decisions. Continued investment should depend on future benefit relative to future maintenance cost (Laney 2017).

### 29.1.4 The Gap Between Model Gain and Business Value

The final two illusions occur near the end of the value chain: from data to model metrics, and from model metrics to business outcomes.

A dataset that improves an offline metric provides a value signal, but not the whole story. First, the gain may be local and replaceable. Another cheaper dataset may deliver the same improvement, which means the marginal value of the original asset is limited. More principled data valuation methods, such as Data Shapley, estimate a dataset's average marginal contribution across many data combinations (Ghorbani and Zou 2019; Jia et al. 2019). Their purpose is to expose substitutability, which a single before-and-after metric cannot show. Second, the evaluation itself may be unreliable. Label errors and train-test contamination can make benchmark improvements misleading; widespread label errors in common test sets have been shown to destabilize model rankings (Northcutt, Athalye and Mueller 2021).

The bridge from model metric to business result is even wider. A one-point accuracy improvement may affect a noncritical slice of traffic, be offset by latency or user trust, or be consumed by deployment and monitoring cost. Production ML value often depends less on the model metric alone and more on the engineering discipline around the data and model system (Sculley et al. 2015). Empirical work supports the link between data-driven decision making and firm performance, but the effect is mediated by organizational capability rather than produced automatically by data itself (Brynjolfsson, Hitt and Kim 2011).

### 29.1.5 Section Summary

This section identified four common errors: treating bytes as value, treating spending as output, treating local metric gains as total value, and treating model metrics as business results. The way out is an explicit multidimensional metric system that measures real contribution, reuse, cost, and risk instead of relying on one convenient proxy.

## 29.2 A Metric System for Data Asset Value

### 29.2.1 Why a Metric System Is Needed

Single proxies fail because data value is multidimensional. The same asset may save labeling cost for one team, support a high-frequency retrieval scenario for another, and reduce compliance risk for a governance team. A useful metric system must satisfy three requirements:

- **Measurable.** Each metric needs a clear calculation boundary and data source.
- **Comparable.** Metrics should support ranking and portfolio management across assets.
- **Decision-oriented.** Metrics must help answer whether to continue investment, prioritize governance, expand reuse, or retire the asset.

This chapter groups metrics into three views: usage-side value, benefit-side value, and risk/cost-side value.

### 29.2.2 Usage Side: Reuse Rate and Coverage

Usage-side value asks how widely and deeply the data is used. An unused asset has little realized value no matter how good it looks internally. A reused asset can create large cumulative value even if each use case receives only a modest benefit.

**Reuse rate** measures how many scenarios, teams, and tasks depend on an asset. It can be approximated by the number of downstream consumers or weighted as reuse events multiplied by scenario importance. High reuse creates leverage: one governance improvement benefits all consumers. Repeated "calls" from downstream teams are a direct market signal for value (Pei 2022).

**Coverage** measures how well the asset spans the target problem space. In pre-training data, coverage may mean topic, domain, and language distribution. In evaluation data, it may mean ability dimensions, difficulty levels, and question types. In a RAG knowledge base, it means coverage of real user query distribution. Low coverage can create systematic long-tail failures that are invisible in demos and amplified in production (Sambasivan et al. 2021).

Reuse rate is horizontal: how many scenarios use the asset. Coverage is vertical: how fully the asset covers each scenario. Assets that score well on both are often organizational data infrastructure.

### 29.2.3 Benefit Side I: Training Gain and Retrieval Hits

Benefit-side value asks what concrete improvement the asset creates.

1. **Training gain** measures the improvement when the data is included in training. The standard method is ablation: hold everything else constant, train with and without the asset, and compare metrics on a fixed evaluation set. Training gain must always be stated relative to a baseline and evaluation set. It is also marginal and substitutable; when necessary, Shapley-style methods can allocate contribution more rigorously (Ghorbani and Zou 2019). Gain is nonlinear: deduplication and pruning may let less data produce more value (Lee et al. 2022; Sorscher et al. 2022).

2. **Retrieval hits** measure whether the data is correctly retrieved and supports correct answers in RAG and similar systems. This is online, on-demand value. It can be measured with hit rate, evidence support rate, citation correctness, and real-query performance. Real user query distributions matter because offline constructed queries often miss production complexity (Thakur et al. 2021; Lewis et al. 2020).

Training gain realizes value by solidifying data into parameters. Retrieval hits realize value by calling external knowledge at request time. The same asset may contribute through both paths.

### 29.2.4 Benefit Side II: Labeling Savings and Risk Reduction

Data also creates value by saving cost and reducing risk.

**Labeling savings** measure how much annotation or acquisition cost the asset avoids. A high-quality seed set can reduce labeling volume through active learning because the model requests labels for the most informative samples (Settles 2009). A reusable annotation guideline and labeled corpus can let new tasks add incremental labels instead of starting over. A real-data seed can also guide synthetic data generation. A practical calculation is: cost to reach the same result without the asset minus actual cost after reuse.

**Risk reduction** measures how much the asset lowers compliance, factuality, or decision risk. A well-sanitized corpus with clear provenance and use boundaries can reduce privacy and licensing risk. An authoritative, traceable knowledge base can reduce incorrect or outdated RAG answers, which is especially valuable in finance, healthcare, and government. Risk value can be approximated as the reduction in expected loss: event probability multiplied by loss per event. Dataset documentation practices, such as Datasheets for Datasets, support this risk reduction by making composition, intended use, and limits explicit (Gebru et al. 2021).

### 29.2.5 Risk and Cost Side: Maintenance Cost and Net Value

Data assets require continuous maintenance. Ignoring that cost is another common valuation blind spot.

**Maintenance cost** is the total cost of keeping an asset usable. It includes storage and compute, incremental collection, parsing, indexing, quality checks, contract maintenance, permission management, compliance review, lineage tracking, and the complexity cost of unmanaged dependencies. High-maintenance, low-benefit assets are often data-layer technical debt (Sculley et al. 2015).

Conceptually, net value can be expressed as:

$$
V_{net} = \big(V_{train} + V_{retrieval} + V_{label} + V_{risk}\big)\times U - C_{maintain}
$$

Here $V_{train}$, $V_{retrieval}$, $V_{label}$, and $V_{risk}$ represent training gain, retrieval value, labeling savings, and risk reduction. $U$ captures usage intensity through reuse and coverage. $C_{maintain}$ is maintenance cost. The expression is not meant to produce exact money in every case; it forces teams to reason about multidimensional benefit multiplied by use, minus ongoing cost.

### 29.2.6 Data Asset Value Metric Table

*Table 29-2: Data asset value metric table*

| Metric | View | Calculation boundary | Main use |
| --- | --- | --- | --- |
| Reuse rate | Usage | Number of reuse scenarios or teams, optionally weighted | Identify core assets and leverage |
| Coverage | Usage | Distribution coverage of the target problem space | Assess long-tail capability and blind spots |
| Training gain | Benefit | Metric delta with and without the data | Quantify model contribution |
| Retrieval hits | Benefit | Hit rate, evidence support rate, citation correctness | Quantify online retrieval value |
| Labeling savings | Benefit | Annotation or acquisition cost avoided | Quantify cost substitution |
| Risk reduction | Risk | Expected-loss reduction | Quantify compliance and factuality value |
| Maintenance cost | Cost | Continuing storage, update, and governance cost | Compute net value and identify negative assets |
| Net value | Integrated | Multidimensional benefit times usage minus maintenance cost | Investment and retirement decisions |

Weights differ by scenario. Pre-training teams may emphasize training gain and coverage. RAG teams may emphasize retrieval hits and maintenance cost. Regulated industries may give risk reduction dominant weight.

### 29.2.7 Section Summary

This section decomposed "is this data valuable?" into measurable dimensions: reuse, coverage, training gain, retrieval hits, labeling savings, risk reduction, maintenance cost, and net value. The next question is how one asset can realize value across many paths.

## 29.3 Asset Reuse Paths

### 29.3.1 Reuse as the Value Multiplier

Data is non-rival. One team using it for training does not prevent another from using it for evaluation or retrieval. This makes reuse the strongest amplifier of data value. Every additional reuse path can add benefit at near-zero marginal copying cost.

Many organizations fail to capture this value. Data is collected for one task, used once, then forgotten. Teams repeatedly collect overlapping data without knowing that similar assets already exist. To unlock reuse, two foundations are required: discoverability and trust from the previous chapters, and explicit recording of reuse paths and benefits.

This section follows five typical large-model reuse paths: pre-training, post-training, RAG, evaluation, and compliance.

![Multi-scenario data asset reuse paths](../../images/part9/ch29_fig01_zh.png)

*Figure 29-1: Data asset reuse paths*

### 29.3.2 Pre-training Reuse

Pre-training is the largest reuse path. Data shapes base language ability and domain knowledge, and value is realized through training gain embedded in model parameters.

For a domain asset, pre-training value depends heavily on quality and deduplication rather than scale alone (Lee et al. 2022; Sorscher et al. 2022). Benefit records should include the asset's share in the training corpus, filtering steps, and observed capability improvements relative to baselines. Because full pre-training is costly, teams often estimate gain through smaller proxy-model experiments and extrapolate carefully.

Pre-training value is long-lived but hard to reverse. Once data affects the base model, it influences many downstream tasks and is difficult to remove. This raises the quality and compliance bar: data with unclear copyright or privacy risk can be amplified through the base model. Pre-training benefit must therefore be evaluated together with compliance reuse.

### 29.3.3 Post-training Reuse

Post-training includes supervised fine-tuning, preference alignment, and related stages. It aligns the base model to tasks, domains, and preferences.

Post-training data is smaller but must be high quality and highly targeted. A small domain instruction or preference set can create significant alignment gain (Gunasekar et al. 2023). Labeling savings are especially important: reusable annotation guidelines and labeled examples can reduce the cost of future post-training tasks, particularly when combined with active learning (Settles 2009).

Benefit records should state which post-training process used the data, what behaviors improved, and how much labeling was avoided. Compared with pre-training, post-training is cheaper and more repeatable, so ablation and Shapley-style contribution analysis are more practical (Ghorbani and Zou 2019; Jia et al. 2019). Value is also more scenario-specific: preference data valuable in one domain may transfer poorly to another.

### 29.3.4 RAG Reuse

RAG keeps data outside model parameters and calls it online as external knowledge. Its value is realized through retrieval hits. The distinctive advantages are freshness and updateability: the knowledge base can evolve without retraining the model (Lewis et al. 2020).

To enter the RAG path, a domain asset must be parsed, cleaned, chunked, indexed, and monitored, as discussed in Chapter 21. Maintenance cost matters because stale knowledge, outdated indexes, and poor chunking erode retrieval value. Benefit records should include supported query scenarios, hit rate under real query distributions, evidence support rate, and citation traceability (Thakur et al. 2021).

RAG and training reuse are complementary. Stable, general knowledge may be moved into post-training. Frequently changing knowledge should usually remain in RAG. Putting volatile information into model parameters creates retraining cost and staleness risk; putting stable core knowledge entirely behind retrieval adds avoidable runtime burden.

### 29.3.5 Evaluation Reuse

Evaluation data is often treated as a byproduct of training data, but a high-quality evaluation set is one of the rarest and most durable assets in a large-model organization.

Its value is not direct model improvement; it is the measuring instrument for all improvements. Without reliable evaluation, training gain and retrieval-hit measurements lose their baseline. Label errors in test sets can distort rankings and make harmful changes appear useful (Northcutt, Athalye and Mueller 2021). Benefit records should capture how many model or knowledge-base decisions the evaluation asset supports, what capabilities and difficulty levels it covers, and whether it blocked regressions from release.

Evaluation reuse has one special constraint: it must not leak into training. Once evaluation data contaminates training data, its value as a measuring instrument can collapse. Evaluation assets therefore require strict lineage isolation and contract governance.

### 29.3.6 Compliance Reuse

Compliance is often not described as reuse, but it creates repeatable value. A corpus with clear provenance, licensing boundaries, anonymization, and use records can pass its compliance work downstream to every reuse path.

If a dataset has been reviewed and documented, pre-training, post-training, RAG, and evaluation consumers can inherit that conclusion instead of repeating review independently (Gebru et al. 2021). Conversely, unclear compliance forces every path to carry its own risk or repeat costly review. In regulated industries, compliance value can determine whether an asset can be used at all; a powerful but legally uncertain dataset may have negative net value.

Benefit records should include review cost avoided, legal and privacy risks reduced, and audit records preserved.

### 29.3.7 Benefit Records and Attribution

The five paths form a value-amplification network. To manage it, organizations need a cross-path benefit ledger for each data asset. The ledger should record pre-training capability improvements, post-training labeling savings and alignment gains, RAG query support and hit rates, evaluation decisions protected, and compliance review cost or risk avoided.

This attribution depends on the infrastructure from Chapters 27 and 28: the data catalog gives each asset a stable identity, lineage records how the asset flows into downstream paths, and contracts define use boundaries. Without those foundations, attribution becomes manual, inaccurate, and unsustainable.

### 29.3.8 Section Summary

This section described five reuse paths. Pre-training is long-lived but hard to reverse. Post-training is small, targeted, and rich in labeling savings. RAG emphasizes freshness and updateability. Evaluation provides the measuring instrument and must avoid leakage. Compliance lets one review serve many downstream paths. Reuse is the value multiplier; attribution makes the multiplier governable.

## 29.4 Valuation Case: Enterprise Domain Knowledge Corpus

### 29.4.1 Case Background

Consider a financial services company that has built an enterprise domain knowledge corpus named `fin_domain_corpus`. It contains internal compliance Q&A and customer-service dialogue, licensed industry research reports, and public financial regulations. After cleaning, deduplication, anonymization, and structuring, it is registered as a data product with a data contract.

The corpus can potentially serve all five reuse paths: improving financial-domain capability in a base model, fine-tuning a financial Q&A assistant, supporting a customer-service RAG system, building financial evaluation sets, and lowering downstream compliance risk. The valuation question is whether the corpus deserves continued investment.

### 29.4.2 Input Cost Accounting

Input cost includes one-time acquisition cost and continuing maintenance cost.

*Table 29-3: `fin_domain_corpus` input costs, annualized where applicable, in RMB 10k units*

| Cost item | Type | Amount | Description |
| --- | --- | --- | --- |
| Report licensing | One-time | 80 | Licensed industry research reports |
| Cleaning and anonymization | One-time | 40 | Deduplication, structuring, PII anonymization |
| Labeling | One-time | 60 | Q&A and preference labeling |
| Compliance review | One-time | 20 | Provenance and licensing review |
| Storage and compute | Maintenance | 15/year | Storage, indexing, retrieval compute |
| Updates and rebuilds | Maintenance | 35/year | Incremental collection, reparsing, index rebuilds |
| Governance | Maintenance | 10/year | Contract maintenance, permissions, lineage |

The one-time cost is about RMB 2 million. Annual maintenance is about RMB 600,000. Continued investment should be judged mainly by future benefits relative to annual maintenance, not by historical acquisition cost.

### 29.4.3 Risk Assessment

The corpus has three internal risks. First, licensed reports have use-boundary risk; using them beyond the license, especially in external model pre-training, may violate terms. Second, anonymized customer-service logs still require continuous checks to prevent residual PII leakage. Third, financial regulations change frequently; stale content retrieved by a RAG system may produce costly compliance advice errors.

The corpus also reduces downstream risk because its provenance, review, anonymization, and documentation are centralized. If expected annual loss from compliance and factuality issues would be RMB 500,000 without the corpus and RMB 150,000 with it, annual risk-reduction value is about RMB 350,000. The key insight is that compliance and freshness maintenance are not only cost items; for this asset, they are major value sources.

### 29.4.4 Cross-Path Benefit Calculation

*Table 29-4: Annual cross-path benefits for `fin_domain_corpus`, in RMB 10k units*

| Reuse path | Main value dimension | Annual benefit | Measurement basis |
| --- | --- | --- | --- |
| Pre-training | Training gain | 30 | Proxy-model financial capability uplift |
| Post-training | Training gain plus labeling savings | 70 | Q&A assistant alignment gain and avoided relabeling |
| RAG | Retrieval hits | 90 | High-frequency customer-service RAG hits reduce agent time |
| Evaluation | Risk reduction | 25 | Version decisions protected and regressions blocked |
| Compliance | Risk reduction | 35 | Shared compliance conclusion lowers expected loss |

The RAG path produces the highest annual benefit because queries are frequent and each hit saves operational work. Post-training follows because the corpus supplies both alignment gain and reusable labels. Pre-training benefit is smaller because the corpus is not large enough to move the base model substantially and licensing boundaries limit use. Evaluation and compliance mainly create risk-reduction value, which is essential in finance even when it does not raise a benchmark metric.

### 29.4.5 Net Value and Decision

Annual cross-path benefit is:

$$
30 + 70 + 90 + 25 + 35 = 250
$$

Annual maintenance cost is 60. Therefore:

$$
V_{net}^{annual} = 250 - 60 = 190 \text{ RMB 10k}
$$

If the one-time acquisition cost of 200 is included, the asset nearly breaks even in year one and contributes about 190 RMB 10k of net value each year afterward. The asset is clearly worth continued investment.

The more important conclusion is qualitative. Its high net value comes from cross-path reuse, not any single path. If the team looked only at pre-training, it might wrongly reject the asset. The maintenance budget for updates and compliance should not be cut, because those costs protect the two largest benefit paths: RAG freshness and compliance risk reduction. The evaluation must also be repeated periodically as regulations, user questions, and model capabilities change.

### 29.4.6 Section Summary

The `fin_domain_corpus` case shows how input costs, risk, cross-path benefits, and maintenance cost combine into net value. The corpus costs about RMB 2 million to acquire and RMB 600,000 per year to maintain, but creates about RMB 2.5 million in annual benefit through five reuse paths. The case also demonstrates why valuation must include reuse, maintenance, and periodic reassessment.

## 29.5 Cost-Benefit Matrix and Portfolio Decisions

### 29.5.1 From Individual Valuation to Portfolio Management

Real organizations manage hundreds or thousands of data assets. Precise valuation for every asset is unnecessary and usually impossible. A more practical approach borrows from portfolio management: classify assets along two axes, value and cost/risk.

The value axis combines benefit-side and usage-side metrics: how much benefit the asset creates and how widely it is reused. The cost/risk axis combines maintenance cost and inherent risk. The result is a cost-benefit matrix.

![Data asset cost-benefit matrix](../../images/part9/ch29_fig02_zh.png)

*Figure 29-2: Cost-benefit matrix*

This matrix supports relative decisions without requiring exact monetary valuation for every asset.

### 29.5.2 Four Quadrant Strategies

1. **High value, low cost: core assets.** These are the best assets: widely reused, high benefit, low maintenance cost, and manageable risk. They deserve protection, continued investment, and active recommendation for reuse.

2. **High value, high cost: heavy-investment assets.** These assets create strong benefits but require expensive updates, review, or governance. The answer is not crude cost cutting, because that may destroy value. Teams should automate maintenance, reuse compliance work, and expand reuse to dilute cost. Only shrink them if cost reduction is impossible and benefit no longer covers cost.

3. **Low value, low cost: long-tail assets.** These assets do not create much current benefit but are cheap to keep. They can remain discoverable with minimal maintenance, because future scenarios may raise their value. They should not consume governance resources needed by core assets.

4. **Low value, high cost: negative assets.** These are the dangerous assets: low benefit with high cost or risk. They often survive because of the cost illusion: "we already spent so much." Correct handling is archive, decommission, or retire, after impact analysis. Cleaning negative assets is often one of the highest-ROI governance actions.

### 29.5.3 Using the Matrix for Portfolio Decisions

The matrix turns separate asset evaluations into resource-allocation decisions. It helps answer which assets deserve governance resources, which assets quietly consume cost without output, and which valuable assets require cost-reduction programs to remain sustainable.

A healthy portfolio protects and reuses core assets, optimizes heavy-investment assets, keeps long-tail assets cheaply discoverable, and retires negative assets. This balance is dynamic; assets move as their use, cost, and risk change.

### 29.5.4 Section Summary

The cost-benefit matrix lifts valuation from individual assessment to portfolio management. It classifies assets as core, heavy-investment, long-tail, or negative assets and ties each class to a management strategy. Because quadrant position changes over time, valuation must become periodic.

## 29.6 Asset Review Cards and Continuous Value Governance

### 29.6.1 Value Decays, So Valuation Must Be Periodic

Data value is not static. Financial regulations become outdated, customer-service knowledge drifts away from real questions, and once-valuable labels may be replaced by better data. A one-time valuation at procurement or launch will drift out of date and eventually reproduce the illusions described earlier.

The lightweight mechanism proposed here is an **asset review card**: a one-page periodic record of value, cost, risk, reuse, and recommended action.

![Data asset review card](../../images/part9/ch29_fig03_zh.png)

*Figure 29-3: Asset review card*

### 29.6.2 Core Elements of a Review Card

A practical card should include five groups of information:

1. **Basic information and ownership.** Asset name, owner, related data product, and contract version.
2. **Value metric snapshot.** Reuse rate, coverage, training gain, retrieval hits, labeling savings, risk reduction, maintenance cost, and net value, compared with the previous review.
3. **Reuse path ledger.** Which of the five reuse paths currently use the asset, what benefit each path creates, and what reuse was added or lost since the previous cycle.
4. **Matrix position and action recommendation.** Current cost-benefit quadrant and corresponding action: maintain, optimize, keep cheaply, or retire.
5. **Risks and next actions.** Freshness, compliance, quality, or ownership risks, plus concrete tasks before the next review.

### 29.6.3 Value Decay and Review Cadence

The review cadence should match the speed of value decay. Highly time-sensitive assets, such as financial regulations, news corpora, and customer-service knowledge bases, require frequent reviews focused on RAG hit rate and staleness. Stable assets, such as foundational language corpora or durable domain knowledge, can be reviewed less frequently. Evaluation assets should be reviewed after major model or knowledge-base releases, with special attention to validity and leakage risk.

Key decay signals include falling reuse rate, falling retrieval hit rate, rising maintenance cost with flat benefit, and compliance status changes. These signals should trigger deeper evaluation and possible movement between matrix quadrants.

### 29.6.4 Section Summary

The asset review card turns valuation into a continuing governance mechanism. It compresses metrics, reuse paths, portfolio position, risk, and actions into a periodic checklist. Continuous review is what keeps value governance aligned with reality.

## 29.7 Engineering the Valuation Process

### 29.7.1 From One-Time Calculation to Evaluation Pipeline

The hard part is not understanding valuation concepts; it is embedding them into daily data engineering. If valuation exists only in budget reviews, procurement retrospectives, or project acceptance reports, it will not change behavior. A more useful approach is a lightweight but stable evaluation pipeline that leaves value signals whenever data is created, changed, called, reused, or retired.

The pipeline has six stages:

1. **Asset registration.** Record identity, owner, source boundary, compliance status, quality level, version, and intended consumption scenarios.
2. **Usage collection.** Capture logs from training references, retrieval hits, dashboard subscriptions, feature use, evaluation-set use, and downstream jobs.
3. **Benefit estimation.** Start with proxies such as labeling samples saved, manual work avoided, task success improved, or expected incidents reduced.
4. **Cost aggregation.** Separate one-time acquisition and cleaning cost from future storage, update, review, and maintenance cost.
5. **Value scoring.** Combine usage, benefit, cost, and risk into comparable grades or net-value estimates.
6. **Decision feedback.** Write results back into recommendation, investment, maintenance-frequency, archive, retirement, and governance workflows.

Without decision feedback, the pipeline is only another monitoring system.

### 29.7.2 Asset Registration: The Minimal Fact Table

Many organizations cannot value data because they cannot answer basic questions: what is this asset, who owns it, who uses it, and can it be reused? Registration is the minimal fact table for valuation.

*Table 29-5: Example fields for data asset registration*

| Field | Meaning | Valuation role |
| --- | --- | --- |
| asset_id | Unique data asset identifier | Primary key for attribution and cross-system joins |
| asset_name | Asset name | Search and human recognition |
| owner | Responsible owner | Review, maintenance, and decision accountability |
| domain | Business or knowledge domain | Coverage and cross-domain reuse analysis |
| source_type | Internal, purchased, public, and so on | Licensing, cost, and compliance risk judgment |
| contract_version | Data contract version | Reliability and breaking-change tracking |
| quality_level | Quality grade | Reuse recommendation and risk assessment |
| compliance_status | Compliance state | Eligibility for training, RAG, evaluation, and other paths |
| refresh_frequency | Update cadence | Maintenance cost and freshness-risk estimation |
| expected_scenarios | Intended consumption scenarios | Baseline for later actual reuse |

Mandatory fields can start small: `asset_id`, `owner`, `source_type`, `compliance_status`, and `refresh_frequency`. Registration should also avoid equating physical tables with assets. A data asset may include tables, files, vector indexes, label sets, documentation, and compliance records. The boundary should follow consumption semantics: what stable capability downstream users depend on.

### 29.7.3 Usage Collection: Making Reuse Observable

Reuse rate cannot rely on manual forms forever. The pipeline should collect reuse events from systems:

- Training corpora: training configs, data loading manifests, version lock files, and experiment tracking.
- RAG knowledge assets: retrieval logs, vector-index calls, chunk hits, and answer citations.
- Evaluation assets: evaluation configs, release records, and regression reports.
- Reporting assets: query logs, dashboard visits, subscriptions, and exports.
- Feature assets: online feature calls, offline training references, and model registry records.

Each reuse event should record who used the asset, when, how, which version, and what result was observed.

*Table 29-6: Example reuse event log fields*

| Field | Example | Description |
| --- | --- | --- |
| event_id | reuse_2026_0001 | Unique reuse event identifier |
| asset_id | fin_domain_corpus | Reused asset |
| asset_version | v2026.05 | Asset version |
| consumer | risk_qa_assistant | Downstream consumer |
| reuse_path | RAG | Reuse path |
| event_time | 2026-05-18 10:00:00 | Time of use |
| usage_count | 12836 | Calls or references |
| outcome_metric | hit_rate=0.71 | Outcome proxy |
| attribution_weight | 0.6 | Benefit attribution weight |

`attribution_weight` matters because real outcomes are usually produced by several datasets, models, and engineering components. Early-stage rules can allocate benefit by hit count, sample share, ablation results, or expert judgment. Mature teams can introduce Shapley approximations, experiments, or causal attribution.

### 29.7.4 Benefit Estimation: Coarse First, Precise Later

Benefit estimation often fails by pursuing false precision or refusing to estimate at all. A better path is coarse first, precise later. Coarse does not mean arbitrary; it means stable, explainable, reviewable proxies.

For RAG assets, proxies include retrieval hits, lower human handoff rate, and shorter handling time. For post-training data, proxies include annotation cost avoided, alignment metric improvement, and review pass-rate improvement. For evaluation sets, proxies include blocked regression releases and reduced expected incident loss. For compliant corpora, proxies include review hours saved and lower violation probability.

Benefits can first be graded as high, medium, or low, then monetized for high-benefit or high-cost assets when budget decisions require precision. Each estimate should include `confidence_level`. High confidence comes from A/B tests, real bills, automatic logs, and reproducible experiments. Medium confidence comes from stable historical statistics and cross-team confirmation. Low confidence comes from one-off interviews or subjective assumptions.

### 29.7.5 Cost Aggregation: Do Not Let Bills Hide Engineering Facts

Cost aggregation is an engineering problem as much as a finance problem. Storage cost may come from object stores, warehouses, lakehouse tables, and vector databases. Compute cost may come from schedulers, training jobs, index builds, and batch tasks. Labeling cost may come from labeling-platform records and vendor invoices. Quality cost may come from quality-check jobs and incident tickets. Compliance cost may come from review workflows, legal assessment, licensing, and anonymization pipelines.

Shared infrastructure requires allocation rules. They do not need to be perfect, but they must be stable, transparent, and explainable. Common rules allocate by storage volume, call count, compute time, sample count, or human hours. Critical assets can use separate ledgers.

Cost records should distinguish reducible and irreducible cost. Historical purchases and past labeling are sunk. Future storage, updates, index rebuilds, monitoring, human review, and compliance maintenance are reducible. Retirement decisions should mainly depend on future reducible cost and future benefit.

### 29.7.6 Scoring and Grading: Making Valuation Executable

Valuation must become an actionable grade. One practical system is S/A/B/C/D:

*Table 29-7: Example data asset value grades*

| Grade | Criteria | Management action |
| --- | --- | --- |
| S | High benefit, high reuse, low or controlled risk | Protect, prioritize investment, strong SLA |
| A | Clearly supports important scenarios with stable benefit | Maintain and promote reuse |
| B | Has stable but limited downstream use | Basic maintenance and periodic review |
| C | Weak evidence of value or only long-tail use | Low-cost preservation and observation |
| D | Low benefit with high cost or high risk | Archive, retire, stop new investment |

Grades should not be assigned by the platform team alone. Asset owners, major consumers, business owners, compliance owners, and platform owners should participate for important assets. In particular, D-grade assets should usually enter a retirement observation period before final shutdown, because they may support rare but critical audit or incident-review needs.

### 29.7.7 Decision Feedback: Writing Valuation Back Into Governance

Valuation creates impact only when written back into governance systems. In the catalog, high-value assets should receive better search ranking, clearer examples, and recommendation badges. In permissions, high-value and compliant assets can receive smoother application paths, while high-risk assets receive stronger approval, usage limits, and audit. In quality management, S and A assets should have stricter monitoring and change review than B and C assets. In cost management, D assets should trigger optimization or retirement, while high-value high-cost assets enter targeted cost-reduction plans. In planning, repeated unmet searches should suggest new common assets, while persistent low-value high-cost assets should challenge past data-construction priorities.

Ordinary monitoring asks what happened. Valuation also asks where resources should go next.

### 29.7.8 Engineering Checklist

*Table 29-8: Data valuation engineering checklist*

| Check item | Passing standard |
| --- | --- |
| Asset registration | Key assets have unique ID, owner, source, compliance status, and version |
| Usage collection | At least one reuse event type can be collected automatically from training, RAG, evaluation, or reporting |
| Benefit estimation | Key assets have benefit proxies and confidence notes |
| Cost aggregation | One-time, maintenance, and reducible costs are separated |
| Risk records | Licensing, privacy, freshness, and quality risks are recorded |
| Value grading | Key assets are graded S/A/B/C/D or similar |
| Decision feedback | Results affect catalog recommendation, maintenance priority, or retirement plans |
| Periodic review | High-value and high-risk assets have a defined review cadence |

Early pilots should focus on 10 to 20 high-reuse, high-cost, high-risk, or disputed assets rather than trying to cover everything.

### 29.7.9 Section Summary

This section turned valuation into an engineering process: register assets, collect reuse, estimate benefit, aggregate cost, score value, and feed decisions back into governance. The goal is not exact monetary value on day one, but a growing base of reviewable value facts.

## 29.8 Organizational Collaboration, Governance, and Anti-Patterns

### 29.8.1 Data Value Cannot Be Defined by One Team Alone

Data teams understand production, but not always business benefit. Business teams understand outcomes, but not always lineage, quality, and compliance. Algorithm teams understand model effects, but not always acquisition and maintenance cost. Compliance teams understand risk, but not always downstream reuse.

If any one role defines value alone, the judgment becomes partial. Mature value governance therefore needs cross-role collaboration. It can start with a monthly data asset review focused only on assets with meaningful changes: fast-growing reuse, rising maintenance cost, changed compliance status, long-term nonuse with high cost, or new entry into a core production path. The input is valuation-pipeline facts. The output must be concrete governance action.

### 29.8.2 RACI: Clarifying Responsibility for Value Decisions

RACI stands for Responsible, Accountable, Consulted, and Informed. It reduces ambiguity in value governance.

*Table 29-9: Example RACI for data value governance*

| Work item | R | A | C | I |
| --- | --- | --- | --- | --- |
| Asset registration | Data owner | Data domain lead | Platform team, compliance team | Downstream consumers |
| Usage collection | Platform team | Data platform lead | Algorithm team, application team | Data owner |
| Benefit estimation | Downstream consumers | Business owner | Data owner, algorithm team | Platform team |
| Cost aggregation | Platform team | Data platform lead | Finance team, data owner | Business owner |
| Risk assessment | Compliance team | Compliance lead | Data owner, business team | Platform team |
| Value grading | Data governance committee | Data lead | Business, algorithm, compliance, platform | Related teams |
| Retirement decision | Data owner | Data domain lead | Downstream consumers, compliance team | Platform team |

The table is not universal. The important point is that every action has an accountable owner; otherwise valuation remains a recommendation rather than a decision.

### 29.8.3 Incentives: Making Reuse More Attractive Than Rebuilding

Reuse often loses to rebuilding because incentives point the wrong way. Reusing an asset may require communication, permission requests, contract reading, and field adaptation. Building a private copy can look faster in the short term. Teams are often rewarded for new delivery, not for avoiding duplicate construction. Asset owners carry maintenance responsibility but may not see credit for downstream reuse.

A useful incentive mechanism makes reuse benefits visible to both producers and consumers. Producers should see how many downstream teams reuse their assets, what duplicate work was avoided, and which key scenarios were supported. Consumers should see the development, labeling, review, and maintenance cost they avoided by reusing. The platform can lower friction by showing quality grade, compliance status, example queries, recommended use, known consumers, SDKs, notebooks, RAG templates, evaluation scripts, and contract-change notifications.

### 29.8.4 Anti-Pattern 1: Valuation Without Governance

The first anti-pattern is building metrics, dashboards, and scoring models without connecting them to action. High-value assets receive no better maintenance, low-value assets are not downgraded, high-risk assets are not restricted, and negative assets are not retired. The system becomes another dashboard no one uses.

Avoid this by binding every grade to actions, every action to an owner, and every owner to a deadline. For example, S-grade assets require quality monitoring and change notification; D-grade assets require retirement assessment within one review cycle; high-risk assets require compliance review before entering a new reuse path.

### 29.8.5 Anti-Pattern 2: Overcomplicated Scoring Models

Some organizations begin with dozens of indicators, complex weights, and predictive models. The model looks sophisticated, but inputs are unreliable and weights lack consensus. The result is hard to explain and hard to trust.

Early scoring should be simple and explainable. A five-factor score using number of reuse paths, number of key scenarios, annual maintenance cost, compliance risk, and quality grade is often enough. More complex models can come later. Human review remains necessary: a low-scoring asset may support rare critical audits, and a high-scoring asset may be unusable if compliance boundaries are unclear.

### 29.8.6 Anti-Pattern 3: Ignoring Version Differences

Data assets change. A field change, anonymization-policy update, source replacement, or label-guideline revision can change value. If valuation binds only to the asset name, it mixes benefits and risks from different versions. A RAG knowledge base may have high hit rate in v1 and worse hit rate after a v2 chunking change. A corpus may be internal-research-only in v1 and commercially usable in v2 after licensing expansion. Reuse logs, benefit records, risk records, and cost records should all bind to versions.

### 29.8.7 Anti-Pattern 4: Treating Compliance as Pure Cost

Compliance review is often seen only as a delay. That misses the reuse value of clear provenance, authorization, anonymization, and use boundaries. A compliant asset can enter more downstream scenarios with less repeated review and lower expected loss. In large-model settings, where data can be solidified into model behavior, licensing boundaries, privacy protection, and traceability are even more important than in traditional analytics.

Compliance status should therefore be part of the value metric system. Clear compliance is not just a cost; it is a value multiplier.

### 29.8.8 Anti-Pattern 5: Undervaluing Retirement

Organizations like discussing new data construction and avoid discussing retirement. But retiring low-value high-cost assets is value creation. It saves storage and compute, reduces monitoring and review burden, lowers misuse risk, and makes catalogs easier to navigate.

Retirement must be orderly. It should include impact analysis, downstream notification, an observation period, archive strategy, and recovery plan. Retirement is not merely deletion; it is controlled removal from production-grade dependency.

### 29.8.9 Rollout Paths for Different Maturity Levels

Organizations at different maturity levels should proceed differently:

- **Early stage: make assets visible.** Build registration, owner assignment, and basic reuse records. Do not chase fine-grained monetization first.
- **Middle stage: make assets comparable.** Add unified grades, cost aggregation, and review mechanisms so teams can prioritize governance.
- **Advanced stage: optimize the portfolio.** Introduce benefit attribution, online experiments, automatic cost allocation, portfolio optimization, and cross-domain reuse recommendation.

The stages can be summarized as visible, comparable, and optimizable. Rollout should not be packaged as one large platform transformation. Start with one domain, a few high-frequency assets, and real downstream consumers. Run the registration, collection, evaluation, review, and action loop. Then turn successful fields, metrics, and workflows into platform capability.

### 29.8.10 Section Summary

Data value governance is organizational, not merely analytical. It requires shared judgment across data, business, algorithm, platform, finance, and compliance roles. RACI clarifies responsibility, incentives make reuse rational, version governance keeps valuation accurate, compliance becomes a value enabler, and retirement keeps the portfolio healthy. Valuation closes the loop only when it affects catalog ranking, access approval, quality monitoring, budget allocation, and asset retirement.

## Chapter Summary

This chapter explained how to value data scientifically and amplify value through deliberate reuse.

Data value is often misjudged because teams substitute easy proxies for real value: bytes for value, spending for output, local model gain for asset value, and model metrics for business results. Research on scaling laws, compute-optimal training, deduplication, pruning, Data Shapley, and evaluation reliability shows that value depends on information density, quality, relevance, and comparable contribution, not simply on size or cost (Kaplan et al. 2020; Hoffmann et al. 2022; Lee et al. 2022; Ghorbani and Zou 2019).

The chapter built a multidimensional metric system covering reuse rate, coverage, training gain, retrieval hits, labeling savings, risk reduction, maintenance cost, and net value. It then showed how one data asset can be reused across pre-training, post-training, RAG, evaluation, and compliance, with benefits recorded and attributed back through catalogs, lineage, and contracts.

The financial-domain corpus case demonstrated that valuation is practical. A corpus with about RMB 2 million in acquisition cost and RMB 600,000 in annual maintenance can create about RMB 2.5 million in annual cross-path benefit, mostly because it is reused and maintained. The cost-benefit matrix extends individual valuation into portfolio management. Asset review cards make valuation periodic.

Finally, the chapter argued that valuation must be engineered and governed. Asset registration, usage collection, benefit estimation, cost aggregation, grading, and decision feedback turn data value into a manageable object. RACI, reuse incentives, version governance, compliance reuse, and retirement mechanisms ensure the metrics change behavior. Together with catalogs, products, and contracts, valuation completes the shift from passive data resource to active data asset.

## References

Brynjolfsson E, Hitt L M, Kim H H (2011) Strength in Numbers: How Does Data-Driven Decisionmaking Affect Firm Performance? Available at SSRN 1819486.

Fleckenstein M, Obaidi A, Tryfona N (2023) A Review of Data Valuation Approaches and Building and Scoring a Data Valuation Model. Harvard Data Science Review 5(1).

Gebru T, Morgenstern J, Vecchione B, Vaughan J W, Wallach H, Daume III H, Crawford K (2021) Datasheets for Datasets. Communications of the ACM 64(12):86-92.

Ghorbani A, Zou J (2019) Data Shapley: Equitable Valuation of Data for Machine Learning. In: Proceedings of the 36th International Conference on Machine Learning (ICML), pp 2242-2251.

Gunasekar S, Zhang Y, Aneja J, Mendes C C T, Del Giorno A, Gopi S, Javaheripi M, Kauffmann P, de Rosa G, Saarikivi O, Salim A, Shah S, Behl H S, Wang X, Bubeck S, Eldan R, Kalai A T, Lee Y T, Li Y (2023) Textbooks Are All You Need. arXiv preprint arXiv:2306.11644.

Hoffmann J, Borgeaud S, Mensch A, Buchatskaya E, Cai T, Rutherford E, de Las Casas D, Hendricks L A, Welbl J, Clark A, Hennigan T, Noland E, Millican K, van den Driessche G, Damoc B, Guy A, Osindero S, Simonyan K, Elsen E, Rae J W, Vinyals O, Sifre L (2022) Training Compute-Optimal Large Language Models. In: Advances in Neural Information Processing Systems 35.

Jia R, Dao D, Wang B, Hubis F A, Hynes N, Gurel N M, Li B, Zhang C, Song D, Spanos C J (2019) Towards Efficient Data Valuation Based on the Shapley Value. In: Proceedings of the 22nd International Conference on Artificial Intelligence and Statistics (AISTATS), pp 1167-1176.

Kaplan J, McCandlish S, Henighan T, Brown T B, Chess B, Child R, Gray S, Radford A, Wu J, Amodei D (2020) Scaling Laws for Neural Language Models. arXiv preprint arXiv:2001.08361.

Laney D B (2017) Infonomics: How to Monetize, Manage, and Measure Information as an Asset for Competitive Advantage. Routledge, New York.

Lee K, Ippolito D, Nystrom A, Zhang C, Eck D, Callison-Burch C, Carlini N (2022) Deduplicating Training Data Makes Language Models Better. In: Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (ACL), pp 8424-8445.

Lewis P, Perez E, Piktus A, Petroni F, Karpukhin V, Goyal N, Kuttler H, Lewis M, Yih W-t, Rocktaschel T, Riedel S, Kiela D (2020) Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. In: Advances in Neural Information Processing Systems 33, pp 9459-9474.

Moody D, Walsh P (1999) Measuring the Value of Information: An Asset Valuation Approach. In: Proceedings of the 7th European Conference on Information Systems (ECIS), pp 496-512.

Northcutt C G, Athalye A, Mueller J (2021) Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks. In: Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks.

Pei J (2022) A Survey on Data Pricing: From Economics to Data Science. IEEE Transactions on Knowledge and Data Engineering 34(10):4586-4608.

Sambasivan N, Kapania S, Highfill H, Akrong D, Paritosh P, Aroyo L M (2021) "Everyone wants to do the model work, not the data work": Data Cascades in High-Stakes AI. In: Proceedings of the 2021 CHI Conference on Human Factors in Computing Systems, pp 1-15.

Sculley D, Holt G, Golovin D, Davydov E, Phillips T, Ebner D, Chaudhary V, Young M, Crespo J-F, Dennison D (2015) Hidden Technical Debt in Machine Learning Systems. In: Advances in Neural Information Processing Systems 28, pp 2503-2511.

Settles B (2009) Active Learning Literature Survey. Computer Sciences Technical Report 1648, University of Wisconsin-Madison.

Sorscher B, Geirhos R, Shekhar S, Ganguli S, Morcos A S (2022) Beyond Neural Scaling Laws: Beating Power Law Scaling via Data Pruning. In: Advances in Neural Information Processing Systems 35, pp 19523-19536.

Thakur N, Reimers N, Ruckle A, Srivastava A, Gurevych I (2021) BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models. In: Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks.

Zha D, Bhat Z P, Lai K-H, Yang F, Jiang Z, Zhong S, Hu X (2023) Data-centric Artificial Intelligence: A Survey. arXiv preprint arXiv:2303.10158.
