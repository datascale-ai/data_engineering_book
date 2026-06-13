# Chapter 4: Data Sources, Acquisition, and Copyright

<div class="chapter-authors">Ke Wang</div>

## Abstract

This chapter examines source governance for text pre-training data. It answers three practical questions: which data may be collected, how it should be collected, and how the team can later prove the origin and license boundary of each batch. We first explain why source selection determines model capability, copyright exposure, and the ceiling of downstream cleaning. We then establish a source taxonomy covering open web pages, forums and Q&A, encyclopedic knowledge, code, academic papers, books, enterprise data, and user feedback. The chapter then turns to production ingestion: distributed crawling, heterogeneous parsing, metadata provenance, resumable jobs, and reliability. Finally, it introduces whitelist, graylist, blacklist, and license-classification mechanisms, followed by two anonymized composite cases on Common Crawl processing and internal financial-document governance. After this chapter, readers should be able to build an auditable source inventory, license-decision framework, and metadata standard before large-scale crawling or internal-data ingestion begins.

## Keywords

Data sources; data acquisition; copyright license; Common Crawl; provenance metadata; robots.txt; license classification; source governance

## Learning Objectives

- Distinguish the quality value, scale potential, and license risk of major pre-training data sources.
- Design an ingestion pipeline that includes robots.txt checks, parser-quality review, and resumable processing.
- Record source, license, parser version, processing configuration, and ownership metadata for every data batch.
- Reduce copyright risk with whitelist, graylist, blacklist, and license-classification mechanisms.
- Explain why source governance determines the upper bound of later cleaning and training-data quality.

## Opening Scenario: Why a Team With "Enough Data" Still Failed

The following anonymized composite case uses approximate figures to illustrate engineering scale and risk. An AI research institute spent three months crawling public web resources and accumulated more than 2 TB of Chinese text. By a Chinchilla-style compute/data estimate, this appeared sufficient for one pre-training run of a 7B model. The team launched a two-week training job costing hundreds of thousands of RMB in compute. The first evaluation was disappointing: the model underperformed comparable open-source baselines on Chinese reading comprehension and mathematical reasoning. Worse, it frequently generated SEO-style filler paragraphs and fragments that resembled web-novel forum content.

The training run itself was healthy. The problem was the data. After review, the team found that nearly 60% of the 2 TB corpus came from SEO content farms, roughly 15% came from web novels with uncertain copyright status, and less than 25% contained dense knowledge such as encyclopedic articles, technical documents, or academic summaries. In effect, the team had not collected "data"; it had collected web noise. The model faithfully learned the distribution it was given, producing fluent but hollow text.

This case reveals a basic rule of pre-training data engineering: **source quality determines the ceiling that later cleaning can reach**. If the source is low-quality by nature, even a sophisticated cleaning pipeline can only remove some noise; it cannot create missing knowledge density or retroactively repair license uncertainty.

---

## 4.1 Why Pre-training Corpora Can Fail at the Source

### 4.1.1 Source Selection Sets the Capability Ceiling

Pre-training teaches a language model to "read" large-scale text and form a statistical map of language, facts, and implicit reasoning patterns. At this stage the model is not learning an explicit rulebook. It is fitting the distribution of the training corpus. Therefore, the model tends to generate what the corpus teaches it to generate.

The implication is direct: **the quality ceiling of pre-training data is the capability ceiling of the model**. A model pre-trained on diverse, high-density sources can often become useful with a modest amount of SFT data. A model pre-trained on biased or low-quality sources is much harder to repair later, because SFT mostly changes behavior and interaction style; it cannot inject a missing foundation of world knowledge, reasoning structure, or domain coverage that the base model never learned.

### 4.1.2 Three Classic Source-Selection Mistakes

Postmortems of failed pre-training projects repeatedly expose three patterns.

**Biased sources.** The corpus is dominated by one platform, one population, or one writing style. A team may use a leading content platform because the content looks polished, only to find that the resulting model is excellent at that platform's voice and weak at academic writing, legal texts, technical manuals, or other registers. The model develops a strong "platform accent."

**Low-density sources.** The corpus is large but information-poor. SEO farms are the most obvious example. Another common example is direct crawling of repost-heavy social feeds. Individual sentences may be grammatical, but the content is mostly emotional fragments, duplicated slogans, or low-information commentary. Such material is a poor substrate for long-term memory and knowledge extraction.

**High copyright-risk sources.** Some data is large and high-quality but legally dangerous. From 2023 to 2024, OpenAI, Google, Stability AI, and others faced lawsuits from publishers, media organizations, and authors over training-data copyright. In China, the Interim Measures for Generative AI Services also require compliant training data and place responsibility for data legality on providers. Ignoring this risk can create legal and commercial exposure long after the model ships.

### 4.1.3 The Misunderstanding That "More Data Is Always Better"

"More data is better" is one of the most common but most easily abused ideas in pre-training. At the macro level it is partly true: if quality is held constant, scale improves capability. At the engineering level it often becomes a substitute for judgment.

The FineWeb paper (Penedo et al. 2024) offers an important observation: under the same token-count constraint, high-quality web corpora filtered from Common Crawl can train stronger models, and data cleaning, deduplication, and source recipes significantly affect final results. DCLM/DataComp-LM further organizes this question into a comparable data-recipe competition: under a fixed training budget, different data filtering and mixing strategies lead to noticeably different downstream performance (Li et al. 2024). In other words, "less but refined" can outperform "more but mixed" in pretraining data, but the conclusion must be tied to the experimental setting, model scale, and evaluation set.

This conclusion lays the foundation for the source-selection strategy throughout the chapter: **the formulation of a data recipe should prioritize each source's knowledge density and information diversity rather than raw volume.**

---

## 4.2 Source Map and Mixing Strategy

If this chapter is an audit checklist for LLM data engineering, the source map is the central view. Before collecting anything, the team should answer: where does the corpus come from, how much does each source contribute, and what are the quality and legal risks?

![Figure 4-1: Layered map of pre-training data sources](../../images/part2/pretrain_data_source_map.png)

*Figure 4-1: Layered map of pre-training data sources. The three-layer taxonomy positions mainstream sources by processing complexity, knowledge density, and license risk, with typical reference ranges for mixing. Source: original illustration from this book; Alt text: layered map of pretraining data sources showing the quality and compliance positions of open web, forums and Q&A, encyclopedias, code, academic papers, books, enterprise internal data, and user feedback data.*

### 4.2.1 Eight Core Source Categories

**Open web** is the largest and most difficult source category, represented by Common Crawl. Since 2008, Common Crawl has continuously crawled the web and released monthly snapshots containing billions of pages. Its accumulated scale is beyond the petabyte level, and many large pre-training corpora use it as an upstream source. Raw web quality varies dramatically: according to FineWeb, a limited portion of raw Common Crawl is dense body text, while large portions are navigation, ads, SEO spam, JavaScript, and boilerplate. Strict cleaning is mandatory.

**Forums and Q&A** include Reddit, StackOverflow, Zhihu, Quora, and similar platforms. Their value is natural interaction around real questions: follow-ups, corrections, debate, and community explanations. This is useful for conversational ability and question-understanding. StackOverflow remains especially important for code and technical reasoning. The practical caveat is access: many platforms tightened or monetized APIs in 2023-2024, making acquisition more difficult.

**Encyclopedias and structured knowledge** include Wikipedia, Wikidata, and domain wikis. Wikipedia is a near-universal component of pre-training mixtures because it is multilingual, dense, and relatively fact-oriented. It may only represent 1-3% of a training corpus by volume, yet it often contributes disproportionately to factual reliability.

**Code** comes from open-source repositories on GitHub/GitLab and from curated datasets such as The Stack. Code data improves code generation and often transfers positively to natural-language reasoning, likely because code forces structured logic. The Stack is valuable because it organizes hundreds of languages and provides license-filtered versions that retain permissive licenses such as MIT and Apache-2.0.

**Academic papers** include arXiv, PubMed Central, and Semantic Scholar-derived collections. Their knowledge density is extremely high, especially for science, medicine, mathematics, and engineering. arXiv is accessible through an API, but commercial training use still requires careful license review because paper-level licenses differ and have tightened over time.

**Books** offer high language quality, long-context structure, narrative continuity, and deep knowledge. They also carry the highest copyright risk. The Books3 subset in The Pile triggered multiple lawsuits, including litigation involving Meta's LLaMA. Safer approaches include Project Gutenberg for public-domain works or direct licensing from publishers.

**Enterprise proprietary data** is the key differentiator for vertical models. Technical documents, knowledge bases, support tickets, compliance manuals, and SOPs are often dense, structured, and highly relevant. They also contain trade secrets and internal privacy risks, so they require authorization, permission review, and PII removal before training use.

**User feedback and online interaction** form the core of a data flywheel after a model is deployed. Real user conversations, corrections, dissatisfaction signals, and preference labels can feed SFT or RLHF. These records are scarce and valuable, but they are also highly privacy-sensitive and must be governed by consent and redaction.

### 4.2.2 Source Type, License, and Risk Matrix

In practical engineering decisions, source selection cannot be based on quality alone; license risk and acquisition feasibility must also be incorporated into the framework. The following is a risk-profile matrix for major data sources:

*Table 4-1: Source type, license, and risk matrix. Source: compiled by the authors; license risk should be based on specific source terms, robots.txt, service agreements, and legal-review conclusions.*

| Source type | Representative sources | License pattern | Commercial risk | Knowledge density | Scale potential |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Open web | Common Crawl, RefinedWeb | CC-BY / mixed page licenses | Medium: page-specific | Low to medium | PB-level |
| Forums and Q&A | Reddit, StackOverflow, Zhihu | Platform terms | Medium: APIs tightened | Medium | TB-level |
| Encyclopedias | Wikipedia, Wikidata | CC-BY-SA 4.0 | Low | High | Hundreds of GB |
| Open-source code | GitHub, The Stack | MIT / Apache-2.0 after filtering | Low with license filtering | High | Several TB |
| Academic papers | arXiv, PubMed | CC-BY / OA with restrictions | Medium: paper-level review | Very high | Hundreds of GB |
| Copyright books | Books3, Z-Library | Protected by default | High | Very high | Hundreds of GB |
| Public-domain books | Project Gutenberg, Archive.org | Public domain | Very low | High | Several GB |
| Enterprise data | Knowledge bases, docs, tickets | Private authorization | Low after internal approval | Very high | Project-specific |
| User conversations | Product feedback, logs | Consent/privacy terms | Medium: PII-sensitive | High | Product-specific |

### 4.2.3 From Business Goal to Data Recipe

Data mix ratio is one of the most strategic decisions in pretraining data engineering. There is no universal "golden recipe," because different business objectives require different data combinations. The following are reference mixing strategies for four typical business objectives:

*Table 4-2: Data mix strategy by business objective. Source: compiled by the authors; mixing recommendations are a strategic framework and should be calibrated in production through proxy-model evaluation and ablation experiments.*

| Business objective | General web | Code | Academic papers | Books / encyclopedia | Vertical data | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **General Chinese base model** | High | Medium | Low-medium | Medium | Low | Pursues broad knowledge coverage; code should not be too low because it affects reasoning capability. |
| **Code / technical specialized model** | Medium | High | Low-medium | Low-medium | Low | Code proportion rises sharply, but enough general language understanding capability must be retained. |
| **Vertical industry model (e.g., finance/medicine)** | Medium | Low | Medium | Medium | High | Domain data proportion rises significantly, while general corpora preserve baseline general capability. |
| **Multilingual base model** | High | Medium | Low-medium | Low-medium | Allocated by target language | Language distribution in web data must be controlled to match target language-capability requirements. |

Table 4-2 uses "high/medium/low" rather than fixed percentages to avoid misreading one project's experimental recipe as a universal rule. Mixing strategy also needs a **dynamic adjustment mechanism**: different training stages (early pretraining vs. cooldown) should use different mixing weights. The closer training gets to its later stage, the more it should raise the weight of high-quality selected data (books, academic papers, enterprise data) and reduce the weight of low-quality massive data (raw web pages). The LLaMA 3 technical report discloses 15T-scale training data and multi-stage post-training flows (Grattafiori et al. 2024), but it does not provide a directly reusable complete data recipe; production projects must still calibrate through small-model ablations and frozen evaluation sets.

---

## 4.3 Ingestion Pipelines, Parsing, and Provenance

After the source strategy is set, the engineering question becomes: how do we ingest scattered data efficiently, reliably, and legally, while preserving proof for every record?

### 4.3.1 Distributed Asynchronous Crawling and robots.txt Compliance

For tens of millions of URLs, a single-threaded crawler is insufficient. Production systems usually use distributed asynchronous crawling based on `aiohttp`, `Scrapy`, Ray, or Spark. The scheduler must enforce robots.txt checks before request dispatch, both to reduce legal risk and to avoid stressing source sites.

Listing 4-1 gives a lightweight asynchronous concurrent ingestion framework based on `aiohttp`. It uses `urllib.robotparser` to automatically check compliance before sending requests; production environments should also add rate limiting, audit logs, exception retries, and legally maintained source policies.

*Listing 4-1: Example code for asynchronous concurrent ingestion and robots.txt checks. Production environments should add rate limiting, failed retries, audit logs, and source-policy whitelists.*

```python
import asyncio
import aiohttp
from urllib.robotparser import RobotFileParser
from urllib.parse import urlparse


class AsyncEthicalCrawler:
    def __init__(self, user_agent: str = "LLMDataBot/1.0"):
        self.user_agent = user_agent
        self.rp_cache: dict[str, RobotFileParser] = {}

    async def fetch_robots(self, session: aiohttp.ClientSession, domain: str) -> None:
        robots_url = f"https://{domain}/robots.txt"
        rp = RobotFileParser()
        try:
            async with session.get(robots_url, timeout=5) as response:
                if response.status == 200:
                    rp.parse((await response.text()).splitlines())
        except Exception:
            # Production systems should define a legal default here.
            pass
        self.rp_cache[domain] = rp

    async def fetch_url(self, session: aiohttp.ClientSession, url: str) -> str | None:
        domain = urlparse(url).netloc
        if domain not in self.rp_cache:
            await self.fetch_robots(session, domain)

        if not self.rp_cache[domain].can_fetch(self.user_agent, url):
            print(f"Skipping {url}: disallowed by robots.txt")
            return None

        try:
            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    return await response.text()
        except Exception as exc:
            print(f"Failed to fetch {url}: {exc}")
        return None

    async def crawl_batch(self, urls: list[str], concurrency: int = 50) -> list[str | None]:
        sem = asyncio.Semaphore(concurrency)
        async with aiohttp.ClientSession(headers={"User-Agent": self.user_agent}) as session:
            async def bounded_fetch(url: str) -> str | None:
                async with sem:
                    return await self.fetch_url(session, url)

            return await asyncio.gather(*(bounded_fetch(url) for url in urls))
```

This design can raise single-machine throughput to hundreds of QPS while preventing accidental access to disallowed paths. Production systems should add rate limits, audit logs, exception retries, and legal-policy controls.

### 4.3.2 Parsing Heterogeneous Sources

Different sources require different parsing routes. Using the wrong parser either loses valuable content or introduces large amounts of noise.

**HTML and WARC.** Common Crawl provides WARC, WAT, and WET files. WET appears convenient because it contains extracted plain text, but this convenience is a trap for production training. Generic WET extraction often retains navigation, footers, ads, and code fragments. A better route is to parse WARC responses with a high-quality body extractor such as Trafilatura, then evaluate parser yield by language and source.

Listing 4-2 shows an illustrative flow for parsing body text from WARC files while preserving source metadata.

*Listing 4-2: Example code for WARC body parsing and source metadata preservation. This snippet shows the parsing path; production environments should add encoding detection, anomalous-sample isolation, and parsing-quality spot checks.*

```python
import gzip
import trafilatura
from warcio.archiveiterator import ArchiveIterator


def parse_warc_to_clean_text(warc_path: str) -> list[dict]:
    records = []
    opener = gzip.open if warc_path.endswith(".gz") else open
    with opener(warc_path, "rb") as stream:
        for record in ArchiveIterator(stream):
            if record.rec_type != "response":
                continue
            url = record.rec_headers.get_header("WARC-Target-URI")
            try:
                html = record.content_stream().read().decode("utf-8", errors="ignore")
            except Exception:
                continue
            text = trafilatura.extract(
                html,
                url=url,
                include_comments=False,
                favor_precision=True,
                output_format="txt",
            )
            if text and len(text) > 200:
                metadata = trafilatura.extract_metadata(html)
                records.append({
                    "url": url,
                    "text": text,
                    "title": metadata.title if metadata else None,
                    "date": metadata.date if metadata else None,
                    "char_count": len(text),
                    "source": "common_crawl",
                    "warc_file": warc_path,
                })
    return records
```

**PDFs.** Academic papers, books, and enterprise documents are harder because PDF is a page-layout format rather than a semantic-text format. Simple extraction with `pdfplumber` or `PyMuPDF` often interleaves columns in academic papers. Scientific papers may require GROBID, Nougat, or Mathpix; enterprise PDFs should receive manual sampling after parsing to verify paragraph and table structure.

**Git repositories.** Code should normally be obtained by cloning repositories rather than through partial API pulls. The parser should identify language by extension and content, filter generated or very large files, validate syntax where possible, and read license files. For Python, AST parsing can reject corrupted files; for repository-level governance, permissive license whitelists such as MIT and Apache-2.0 are essential.

### 4.3.3 Provenance Metadata: The Birth Certificate of Each Record

Traceable metadata is the foundation of data governance. Without it, a team cannot prove the legality of a record during an audit. "We remember it was legal" is not an audit trail.

Every ingested batch should write metadata at the same time it writes raw data to object storage. The following example fields should be extended according to source type, authorization method, and audit requirements.

Each ingested batch should write standard fields to the metadata database at the same time it lands in object storage. Listing 4-3 gives example fields; actual systems should extend them according to data source, authorization method, and audit requirements.

*Listing 4-3: Example metadata provenance fields for an ingestion batch. Field values are illustrative examples; production environments should extend them according to authorization method, audit requirements, and data-source type.*

```json
{
  "ingestion_id": "cc-2024-10-zh-batch-0042",
  "source_name": "common_crawl_2024_10",
  "source_url": "s3://commoncrawl/crawl-data/CC-MAIN-2024-10/...",
  "ingestion_time": "2024-10-15T08:23:41+08:00",
  "license_type": "cc-crawl-mixed",
  "license_risk": "medium",
  "language": "zh",
  "raw_doc_count": 4280350,
  "raw_size_bytes": 18432000000,
  "parser_version": "trafilatura==1.6.3",
  "filter_config": "min_len=200,favor_precision=true",
  "s3_prefix": "s3://my-bucket/raw/cc-2024-10-zh/",
  "team_contact": "data-team@company.com"
}
```

![Figure 4-2: Data ingestion and provenance chain](../../images/part2/data_ingestion_provenance_chain.png)

*Figure 4-2: Data ingestion and provenance chain. From source contact to final archive, each processing stage appends metadata records to the "Provenance Ledger," forming a complete auditable data-lineage chain. Source: original illustration from this book; Alt text: data ingestion and provenance chain diagram showing the links among source contact, acquisition, parsing, cleaning, storage, and audit records.*

### 4.3.4 Resumability and Job Reliability

Large ingestion jobs often run for days. Node failures, network interruptions, and expired cloud tokens are normal events. Without resumability, every interruption forces a full rerun.

A robust design splits work into moderately sized checkpoint units, often one WARC file per task. After each file is processed, the state database records completion. When the job restarts, completed files are skipped. Combined with Ray Data or Spark fault tolerance, this reduces manual intervention in large-scale ingestion to near zero.

---

## 4.4 Copyright, Licensing, and Traceable Governance

Copyright is one of the most underestimated risks in LLM data engineering. Many teams still assume that "everyone does it this way." That assumption is increasingly unsafe as regulation and litigation mature.

### 4.4.1 Three License Categories

**Open licenses** are relatively safe sources, including CC0, CC-BY, and CC-BY-SA. Wikipedia uses CC-BY-SA 4.0; Project Gutenberg content is generally public domain. These sources may require attribution in model reports or documentation, but they have fewer commercial restrictions. CC-BY-NC is different: the non-commercial clause generally excludes commercial products.

**Commercial licenses** are purchased through explicit agreements with publishers, media organizations, or data vendors. They are usually the safest legally, but expensive and full of specific clauses: a license may apply only to one model version, forbid derivative training data, or restrict redistribution. Legal review must inspect each agreement, and vendors should be required to provide source proof and copyright inventories.

**Gray-zone data** has unclear or contested status. Examples include crawled web text, third-party datasets that package copyrighted books, and sites that disallow crawling through robots.txt but appear in public dumps. For gray-zone sources, introduce risk tiers and joint review by legal and data teams. Whether to include them depends on company risk tolerance, product context, and jurisdiction.

### 4.4.2 Whitelist, Graylist, and Blacklist Controls

The most practical engineering control is a three-list source-governance system.

**Whitelist.** Sources confirmed by legal review with clear permission. Each entry should record license version, usage restrictions, and last review date. Examples: Wikipedia under CC-BY-SA 4.0 with attribution, The Stack v2 after permissive-license filtering, and Project Gutenberg public-domain works.

**Graylist.** Sources with disputed or restrictive terms. They require case-by-case legal review and a recorded decision before use. Examples include paper-level arXiv data and platform API data governed by changing terms of service.

**Blacklist.** Sources that must not be used, including litigated datasets, domains that explicitly disallow AI training, robots.txt-disallowed sites when policy forbids use, or any source marked "not for AI training." The crawler should block these at the ingestion entrance.

Listing 4-4 shows a simplified implementation:

*Listing 4-4: Example code for copyright blacklist interception at the ingestion entrance. In production, the list should be maintained by legal and data-governance teams, and hit reasons and review time should be recorded.*

```python
# Copyright blacklist: intercept prohibited sources at the ingestion entrance
COPYRIGHT_BLACKLIST_DOMAINS = {
    "nytimes.com",       # explicitly prohibits AI-training use
    "wsj.com",           # explicitly requires paid authorization
    "theguardian.com",   # terms of service updated to prohibit AI training
    # ... continuously updated by the legal team
}


def is_url_allowed(url: str) -> bool:
    from urllib.parse import urlparse

    domain = urlparse(url).netloc.lstrip("www.")
    return not any(domain.endswith(blocked) for blocked in COPYRIGHT_BLACKLIST_DOMAINS)
```

### 4.4.3 Automatic License Classification

For code data, license information is usually stored in `LICENSE` or `LICENSE.md` at the repository root and can be automatically identified through rules or classifiers. Listing 4-5 shows a simplified implementation; production systems should use stricter license-parsing libraries and legal-review workflows.

*Listing 4-5: Example code for automatic license-type classification. This snippet is only used to illustrate rule-based recognition; production environments should use mature license-parsing libraries and retain a manual review chain.*

```python
import re

# Common license keyword recognition (simple version; production use should prefer
# a library such as license-expression)
LICENSE_PATTERNS = {
    "MIT": r"(?i)mit\s+license",
    "Apache-2.0": r"(?i)apache\s+license.*2\.0",
    "GPL-3.0": r"(?i)gnu\s+general\s+public\s+license.*version\s+3",
    "CC-BY-4.0": r"(?i)creative\s+commons.*attribution.*4\.0",
    "CC-BY-NC": r"(?i)creative\s+commons.*non.?commercial",
    "Proprietary": r"(?i)(all rights reserved|proprietary|confidential)",
}

COMMERCIAL_SAFE = {"MIT", "Apache-2.0", "CC0", "Public Domain"}


def classify_license(license_text: str) -> dict:
    for name, pattern in LICENSE_PATTERNS.items():
        if re.search(pattern, license_text):
            return {
                "license": name,
                "commercial_safe": name in COMMERCIAL_SAFE,
                "risk_level": "low" if name in COMMERCIAL_SAFE else "high",
            }
    return {"license": "Unknown", "commercial_safe": False, "risk_level": "high"}
```

---

## 4.5 Case Review and Practical Recommendations

### Case 1: End-to-End Lessons From Common Crawl Chinese Corpus Ingestion (Anonymized Composite Case)

**Project background.** A team planned to extract a batch of high-quality Chinese text from a Common Crawl release as the main pretraining corpus for a general Chinese base model. The following scale, time, and ratio descriptions are instructional engineering estimates used to illustrate the risk difference between WET and WARC parsing routes; actual results depend on crawl batch, language-filtering strategy, parser version, and manual spot-check criteria.

**T+0 (decision day).** The team preliminarily assessed the WET files in that batch and decided that directly using WET would be simplest: after all, WET already contains plain text and avoids the parsing step. They downloaded a WET subset and performed a quick evaluation.

**T+3 (problem discovery).** Data engineers randomly sampled Chinese documents for manual review and found that quality was far below expectations. Many documents contained large amounts of navigation and menu text, such as "Home | About Us | Contact Us | Copyright Notice," and there were also problems with advertisement stuffing, product-description stuffing, and body-text truncation. The proportion of genuinely complete article bodies was insufficient to support production-grade training.

**T+4 (route switch).** The team decided to abandon the WET route and instead start from WARC files, reparsing them with Trafilatura in `favor_precision=True` mode. This increased processing time and CPU cost, but it preserved more complete HTML context for the body-text extractor to judge.

**T+8 (reevaluation).** The Trafilatura route improved results substantially. Manual spot checks showed that both the complete-body ratio and average document length were better than with the WET route. The team therefore kept WARC parsing as the production path and wrote the spot-check results into the parsing-quality baseline for this data source.

**Core lesson.** WET files are a "cheap trap": suitable only for rough experiments, not production-grade training-data preparation. Whether to switch to WARC plus a high-quality parser should be determined jointly by spot-checked quality gains, additional time, and CPU cost, rather than by file-acquisition cost alone.

Three directly applicable practices follow from this case. First, **establish a parsing-quality baseline**: during early ingestion of any new data source, manually annotate a random sample of 500-1,000 documents according to startup-stage experience, expand the sample size according to source heterogeneity and error types, and measure complete-body ratio and average document length to form a parsing-quality baseline for that source. Second, **separate evaluation samples from production samples**: do not use the same quick-experiment data to evaluate final training effects, because experimental-stage processing precision is often lower than production-grade precision. Third, **embed quality snapshots in the pipeline**: each processing node (parsing -> filtering -> deduplication) should automatically output a quality snapshot report after completion, recording statistics such as average document length, short-document ratio, and character-set distribution for the current batch. Engineers can then judge whether node output quality meets expectations without additional manual sampling. This automatic quality-snapshot mechanism is one of the core methods for avoiding black-box pipelines in large-scale data engineering.

### Case 2: Compliance Risk in Financial Enterprise Knowledge-Base Ingestion (Anonymized Composite Case)

**Project background.** A financial group decided to train an internal financial Q&A model based on internal research reports, compliance manuals, product descriptions, and other documents. The data scale was approximately 500 GB (PDF + Word format), covering nearly ten years of accumulated internal documents. The following proportions and scale are used to illustrate risk types and do not represent a public event involving any specific company.

**T+0 (data inventory).** Data engineers obtained the document-directory list from the group's IT department and began batch-parsing PDF files. Engineering progressed smoothly, and within a short period the team completed document parsing and preliminary cleaning, producing a batch of candidate training data.

**T+15 (compliance team intervention).** During a routine risk inspection, the group's compliance team found that the dataset contained many copies of files from third-party organizations, such as regulator websites, rating agencies, and external law firms. These files had been historically deposited in the internal OA system, but their copyrights did not belong to the group itself. Some files even contained statements such as "may not be copied or distributed without permission."

**T+16 (emergency pause).** The legal team urgently halted the training task and required a source review of the dataset. The review found that some documents had unclear copyright ownership or clearly belonged to third parties and therefore had to be removed from the training set.

**T+25 (fix completed).** The data team labeled each document category by source ownership, established an internal copyright inventory, removed all third-party documents with questionable copyright, and supplemented authorization evidence for retained documents. Portions of internal documents that cited regulatory provisions were confirmed by compliance to fall within the scope of "reasonable quotation."

**Core lesson.** Internal enterprise documents are not automatically copyright-owned data. Before ingestion begins, all data sources should undergo systematic copyright-ownership review instead of asking the compliance department to intervene only after engineering is complete; the latter has much higher rework cost. It is recommended to introduce a "source-ownership check" node as the first step in the ingestion pipeline, requiring every file to be labeled as copyright holder / internally created / third-party quotation / unknown source, and confirmed with a signature by the document owner from the business department.

## Chapter Summary

This chapter began from how source quality constrains model capability and established a cognitive framework for the pretraining data-source system. It built a layered map covering eight core data-source categories and provided operational quantitative tools for engineering decisions through the risk matrix (Table 4-1) and mixing-strategy matrix (Table 4-2). In the ingestion-pipeline section, this chapter explained the risk of directly using WET, presented a high-quality WARC parsing implementation based on Trafilatura, and established the provenance standard that "every record has a birth certificate." The copyright-governance section introduced the three-tier whitelist, graylist, and blacklist management mechanism, together with license auto-classification code, providing an implementable compliance-engineering solution for commercial LLM teams. The two cases showed from technical and legal perspectives that source governance is the first quality gate of pretraining data engineering.

In the next chapter, we will build on the raw data collected in this chapter and discuss **Chapter 5: Cleaning, Deduplication, and Decontamination**. Source governance determines the upper bound of the corpus that can enter the cleaning pipeline, while the cleaning pipeline determines which samples eventually enter the training set. Together, the two chapters form the quality gatekeeping system for text pretraining data engineering.

## References

Barbaresi A (2021) Trafilatura: A Web Scraping Library and Command-Line Tool for Text Discovery and Extraction. In: Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics, pp 122-131.

Blecher L, Cucurull G, Scialom T, Stojnic R (2023) Nougat: Neural Optical Understanding for Academic Documents. arXiv preprint arXiv:2308.13418.

Grattafiori A, Dubey A, Jauhri A, Pandey A, Kadian A, Al-Dahle A, Letman A, Mathur A, Schelten A, Vaughan A, others (2024) The Llama 3 Herd of Models. arXiv preprint arXiv:2407.21783.

Joulin A, Grave E, Bojanowski P, Douze M, Jegou H, Mikolov T (2017) FastText.zip: Compressing Text Classification Models. arXiv preprint arXiv:1612.03651.

Lopez P (2009) GROBID: Combining Automatic Bibliographic Data Recognition and Term Extraction for Scholarship Publications. In: Proceedings of the 13th European Conference on Digital Libraries, pp 473-474.

Li J, Zhang Y, Yu H, Ma X, Chen Y, Jiang H, Dang K, Goyal T, Keh S, Sherborn M, others (2024) DataComp-LM: In search of the next generation of training sets for language models. arXiv preprint arXiv:2406.11794.

Penedo G, Kydlíček H, Ben Allal L, Lozhkov A, Mitchell M, Raffel C, von Werra L, Wolf T (2024) The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale. arXiv preprint arXiv:2406.17557.

Yu S, Liu Z, Xiong C (2025) Craw4LLM: Efficient Web Crawling for LLM Pretraining. In: Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics. arXiv preprint arXiv:2502.13347.
