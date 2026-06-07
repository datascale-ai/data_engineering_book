# Chapter 4: Data Sources, Acquisition, and Copyright

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

FineWeb (Penedo et al. 2024) offers an important observation: under the same token budget, a carefully filtered Common Crawl subset produced stronger benchmark results than training on far larger raw Common Crawl data. In other words, "less but cleaner" can beat "more but messy."

This motivates the source strategy for the rest of the chapter: a data recipe should prioritize knowledge density, diversity, and license clarity over raw byte volume.

---

## 4.2 Source Map and Mixing Strategy

If this chapter is an audit checklist for LLM data engineering, the source map is the central view. Before collecting anything, the team should answer: where does the corpus come from, how much does each source contribute, and what are the quality and legal risks?

![Figure 4-1: Layered map of pre-training data sources](../../images/part2/pretrain_data_source_map.png)

*Figure 4-1: Layered map of pre-training data sources. The three-layer taxonomy positions mainstream sources by processing complexity, knowledge density, and license risk, with typical reference ranges for mixing. Source: drawn for this book.*

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

Source decisions must weigh quality, legal risk, and acquisition feasibility together.

**Table 4-1: Source type, license, and risk matrix**

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

Data mix ratio is one of the most strategic decisions in pre-training. There is no universal recipe; different product goals require different mixtures.

**Table 4-2: Data mix strategy by business objective**

| Business objective | General web | Code | Academic papers | Books / encyclopedia | Vertical data | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| General Chinese base model | 60-65% | 15-20% | 5-8% | 10-15% | 0-5% | Broad coverage; code should not be too low because it helps reasoning. |
| Code / technical model | 30-35% | 45-55% | 5-10% | 5-8% | 3-5% | Raise code sharply but keep enough general language. |
| Vertical industry model | 25-30% | 8-12% | 15-20% | 10-15% | 30-40% | Domain data becomes prominent while general data preserves broad capability. |
| Multilingual base model | 55-60% | 15-18% | 8-10% | 8-12% | By target language | Control language distribution to match target capability. |

Mixing should also change over time. Early pre-training may use broader data, while cooldown should increase high-quality selected material such as books, papers, code, math, and enterprise data. Public model reports such as LLaMA 3 and Gemma describe similar late-stage upweighting of curated data.

---

## 4.3 Ingestion Pipelines, Parsing, and Provenance

After the source strategy is set, the engineering question becomes: how do we ingest scattered data efficiently, reliably, and legally, while preserving proof for every record?

### 4.3.1 Distributed Asynchronous Crawling and robots.txt Compliance

For tens of millions of URLs, a single-threaded crawler is insufficient. Production systems usually use distributed asynchronous crawling based on `aiohttp`, `Scrapy`, Ray, or Spark. The scheduler must enforce robots.txt checks before request dispatch, both to reduce legal risk and to avoid stressing source sites.

**Listing 4-1: Asynchronous crawling with robots.txt checks**

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

**Listing 4-2: WARC body extraction with provenance metadata**

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

**Listing 4-3: Example ingestion-batch provenance metadata**

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

*Figure 4-2: Data ingestion and provenance chain. Each step appends metadata to a provenance ledger, forming an auditable lineage from source contact to archive.*

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

**Listing 4-4: Copyright blacklist interception**

```python
COPYRIGHT_BLACKLIST_DOMAINS = {
    "nytimes.com",
    "wsj.com",
    "theguardian.com",
}


def is_url_allowed(url: str) -> bool:
    from urllib.parse import urlparse

    domain = urlparse(url).netloc.lstrip("www.")
    return not any(domain.endswith(blocked) for blocked in COPYRIGHT_BLACKLIST_DOMAINS)
```

### 4.4.3 Automatic License Classification

For code data, license text is usually stored in `LICENSE` or `LICENSE.md` at the repository root. Rules or classifiers can identify common patterns, but production systems should use stricter license parsers and legal review.

**Listing 4-5: Simplified license classification**

```python
import re

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

### Case 1: Lessons From Common Crawl Chinese Corpus Ingestion

**Project background.** A team planned to extract roughly 200 GB of high-quality Chinese text from one Common Crawl release for a general Chinese base model. The sizes, timings, and ratios below are engineering estimates as of 2026-06; actual results depend on crawl release, language filtering, parser version, and manual review criteria.

**T+0, decision day.** The team inspected about 15 TB of compressed WET files and chose the WET route because the text was already extracted. They downloaded a 500 GB compressed subset, corresponding to about 50 million documents, and ran a quick review.

**T+3, problem discovery.** Engineers manually reviewed 500 Chinese documents. About 35% contained navigation and menu text, around 20% were ads or product-description stuffing, about 15% were truncated fragments, and fewer than 30% were complete article bodies.

**T+4, route switch.** The team abandoned WET and reprocessed WARC files with Trafilatura in `favor_precision=True` mode. This required about 80 TB of additional compressed WARC processing and took roughly four days on a 200-core CPU cluster.

**T+8, reevaluation.** The WARC route improved quality substantially. In a second 500-document review, complete body-text rate rose to 82%, and average document length increased from about 300 Chinese characters to about 1,100. The final output contained roughly 12 million effective Chinese documents and about 28 billion characters, yielding around 40% more effective content than the WET route with far better quality.

**Core lesson.** WET is a cheap trap: useful for rough experiments, but not for production training data. The extra cost of WARC plus a high-quality parser is usually justified by the quality gain.

Three practices follow from this case. First, establish a parser-quality baseline for every new source by manually reviewing 500-1,000 sampled documents and tracking complete-body ratio and average length. Second, separate experiment samples from production samples; early quick pipelines are rarely production-grade. Third, embed quality snapshots in each processing node, so parsing, filtering, and deduplication produce automatic reports on document length, short-document ratio, charset distribution, and other signals.

### Case 2: Compliance Risk in a Financial Enterprise Knowledge Base

**Project background.** A financial group decided to train an internal Q&A model from research reports, compliance manuals, product documents, and internal PDFs/Word files accumulated over ten years. The corpus was roughly 500 GB. These numbers illustrate risk categories, not a public incident.

The first ingestion plan treated all internal documents as automatically available because they were stored inside the company's knowledge base. Legal review later found several issues: some reports embedded externally licensed market data; some PDF appendices reproduced third-party analyst materials; some Word templates contained customer names and account identifiers; and some documents were only licensed for internal reading, not for model training.

The remediation plan separated documents into four buckets. Documents created fully in-house with approved use entered the whitelist. Documents containing vendor charts or purchased data entered the graylist pending contract review. Documents with customer identifiers were routed through PII redaction before any training use. Documents whose license explicitly forbade derivative use were blacklisted. The data team also added per-document provenance fields for business owner, permission basis, redaction status, and license reviewer.

The lesson is simple: "internal" does not automatically mean "trainable." Enterprise data often has high domain value, but it also carries confidentiality, third-party-license, and privacy constraints. A usable internal-data pipeline must combine permission review, redaction, and auditable metadata before model training begins.

## Chapter Summary

Source governance is not administrative overhead; it is the first technical boundary of pre-training. Source choice shapes model ability, legal risk, and the maximum value of all downstream cleaning. A production source system should include a source map, data mix strategy, robots.txt-aware ingestion, high-quality parsing, provenance records, resumable execution, whitelist/graylist/blacklist controls, and license classification. Only then can later stages such as cleaning, tokenization, and evaluation work on a trustworthy foundation.

## References

- Barbaresi, A. (2021). Trafilatura: A web scraping library and command-line tool for text discovery and extraction.
- Blecher, L. et al. (2023). Nougat: Neural Optical Understanding for Academic Documents.
- Broder, A. Z. (1997). On the resemblance and containment of documents.
- Dubey, A. et al. (2024). The Llama 3 Herd of Models.
- Lopez, P. (2009). GROBID: Combining automatic bibliographic data recognition and term extraction for scholarship publications.
- Penedo, G. et al. (2023). The RefinedWeb Dataset for Falcon LLM.
- Penedo, G. et al. (2024). FineWeb: Decanting the Web for the Finest Text Data at Scale.
