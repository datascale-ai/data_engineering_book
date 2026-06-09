#!/usr/bin/env python3
"""Audit manuscript citations and external reference metadata.

The script focuses on the current Springer Chinese delivery manuscript under
``docs/zh``. It checks three things:

1. Author-year citations in chapter bodies resolve to same-chapter references.
2. Chapter references are actually cited in the same chapter.
3. Reference entries can be verified through stable external signals when
   possible: arXiv IDs, DOI/Crossref, OpenAlex title search, or reachable URLs.

The external checks are intentionally conservative. A "needs-manual-review"
result does not prove a reference is false; it means the script could not
confirm it strongly enough for publisher handoff.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_DIR = ROOT / "publishing" / "final_review"
DEFAULT_ROOTS = [ROOT / "docs" / "zh"]
USER_AGENT = "data-engineering-book-reference-audit/1.0 (mailto:reference-audit@example.invalid)"

REF_HEADER_RE = re.compile(r"^##\s*(?:参考文献|References)\s*$", re.I)
NEXT_H2_RE = re.compile(r"^##\s+")
YEAR_RE = re.compile(r"\b((?:19|20)\d{2})(?:[a-z])?(?=\b|[).,;:])")
ARXIV_RE = re.compile(r"arXiv(?::|\s+preprint\s+arXiv:)\s*([0-9]{4}\.[0-9]{4,5}(?:v\d+)?|[a-z-]+/[0-9]{7}(?:v\d+)?)", re.I)
DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b")
URL_RE = re.compile(r"https?://[^\s<>)]+")

AUTHOR_TOKEN_RE = r"[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’`.-]+"
CITATION_PATTERNS = [
    re.compile(r"\b(National Institute of Standards and Technology)\s*\(?((?:19|20)\d{2})\)?"),
    re.compile(r"\b(Nait\s+Saada|Jimeno\s+Yepes)\s+et\s+al\.\s*\(?((?:19|20)\d{2})\)?"),
    re.compile(r"\b(Kimi\s+Team|Qwen\s+Team|Gemini\s+Team|Open-Sora\s+Team|Wan\s+Team)\s*\(?((?:19|20)\d{2})\)?"),
    re.compile(rf"\b({AUTHOR_TOKEN_RE})\s+et\s+al\.\s*\(?((?:19|20)\d{{2}})\)?"),
    re.compile(rf"\b({AUTHOR_TOKEN_RE}),\s+{AUTHOR_TOKEN_RE}\s+and\s+{AUTHOR_TOKEN_RE}\s*\(?((?:19|20)\d{{2}})\)?"),
    re.compile(rf"\b({AUTHOR_TOKEN_RE}),\s+{AUTHOR_TOKEN_RE}\s*&\s+{AUTHOR_TOKEN_RE}\s*\(?((?:19|20)\d{{2}})\)?"),
    re.compile(rf"(?<!,\s)\b({AUTHOR_TOKEN_RE})\s+and\s+{AUTHOR_TOKEN_RE}\s*\(?((?:19|20)\d{{2}})\)?"),
    re.compile(rf"(?<!,\s)\b({AUTHOR_TOKEN_RE})\s*&\s*{AUTHOR_TOKEN_RE}\s*\(?((?:19|20)\d{{2}})\)?"),
    re.compile(rf"\b({AUTHOR_TOKEN_RE})\s*等\s*\(?((?:19|20)\d{{2}})\)?"),
    re.compile(rf"\b(NIST|OpenAI|OWASP|GDPR|DVC|MLflow|Kubernetes|MindSpore|Apache|Ray|dbt)\s*\(?((?:19|20)\d{{2}})\)?"),
]

NON_REFERENCE_CONTEXT = {
    "table",
    "figure",
    "fig",
    "chapter",
    "section",
    "part",
    "ch",
    "p",
}

SOURCE_CUES = [
    ". In:",
    ". Proceedings",
    ". In Proceedings",
    ". Advances in",
    ". Journal",
    ". Communications",
    ". Transactions",
    ". Nature",
    ". Science",
    ". IEEE",
    ". ACM",
    ". arXiv",
    ". International Conference",
    ". Conference",
    ". Workshop",
    ". Foundations",
    ". National Institute",
    ". OpenAI",
    ". Hugging Face",
    ". Available at:",
]


@dataclass
class ReferenceEntry:
    file: str
    line: int
    entry_no: int
    entry: str
    first_author: str
    year: str
    title: str
    doi: str
    arxiv: str
    url: str
    key: str
    fingerprint: str
    format_issues: list[str]


@dataclass
class BodyCitation:
    file: str
    line: int
    author: str
    year: str
    text: str
    key: str
    context: str


@dataclass
class ExternalCheck:
    file: str
    line: int
    entry_no: int
    key: str
    title: str
    status: str
    source: str
    matched_title: str
    matched_year: str
    score: float
    identifier: str
    url_status: str
    issues: list[str]


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def book_files(roots: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        for path in sorted(root.rglob("*.md")):
            if "superpowers" in path.parts or path.name in {"index.md", "translation-status.md"}:
                continue
            name = path.name
            if name.startswith("ch") or re.match(r"p\d+", name) or name.startswith("appendix_"):
                files.append(path)
    return files


def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def normalize_word(text: str) -> str:
    text = strip_accents(text)
    text = text.replace("’", "'").replace("`", "'")
    text = re.sub(r"[^A-Za-z0-9]+", "", text)
    return text.lower()


def author_key(author: str) -> str:
    if author == "National Institute of Standards and Technology":
        return "nist"
    key = normalize_word(author)
    aliases = {
        "teamkimi": "kimiteam",
        "teamqwen": "qwenteam",
    }
    return aliases.get(key, key)


def normalize_title(text: str) -> str:
    text = strip_accents(text).lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"arxiv:[^\s.]+", " ", text, flags=re.I)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def similarity(a: str, b: str) -> float:
    a_norm = normalize_title(a)
    b_norm = normalize_title(b)
    if not a_norm or not b_norm:
        return 0.0
    return SequenceMatcher(None, a_norm, b_norm).ratio()


def year_delta_too_large(left: str, right: str) -> bool:
    if not (left and right and left.isdigit() and right.isdigit()):
        return False
    return abs(int(left) - int(right)) > 1


def parsed_year(match: re.Match[str] | None) -> str:
    if not match:
        return ""
    return match.group(1)


def clean_url(url: str) -> str:
    return url.rstrip(".,;。；")


def extract_reference_section(path: Path) -> list[tuple[int, str]]:
    lines = read_text(path).splitlines()
    in_refs = False
    rows: list[tuple[int, str]] = []
    for lineno, line in enumerate(lines, 1):
        stripped = line.strip()
        if REF_HEADER_RE.match(stripped):
            in_refs = True
            continue
        if in_refs and NEXT_H2_RE.match(stripped):
            break
        if not in_refs or not stripped or stripped in {"---", "----"}:
            continue
        stripped = re.sub(r"^[-*]\s+", "", stripped)
        rows.append((lineno, stripped))
    return rows


def body_lines_before_references(path: Path) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    for lineno, line in enumerate(read_text(path).splitlines(), 1):
        if REF_HEADER_RE.match(line.strip()):
            break
        rows.append((lineno, line))
    return rows


def first_author_from_entry(entry: str) -> str:
    entry = re.sub(r"^\[[^\]]+\]\s*", "", entry).strip()
    entry = re.sub(r"^\d+\.\s*", "", entry).strip()
    if entry.startswith("National Institute of Standards and Technology"):
        return "NIST"
    if entry.startswith("Nait Saada"):
        return "Nait Saada"
    if entry.startswith("Jimeno Yepes"):
        return "Jimeno Yepes"
    if entry.startswith("Team Kimi"):
        return "Kimi Team"
    for team_name in ("Qwen Team", "Gemini Team", "Open-Sora Team", "Wan Team"):
        if entry.startswith(team_name):
            return team_name
    if entry.startswith("van den Oord"):
        return "Oord"
    if entry.startswith("OWASP"):
        return "OWASP"
    if entry.startswith("OpenAI"):
        return "OpenAI"
    if entry.startswith("Apache "):
        return "Apache"
    if entry.startswith("Ray "):
        return "Ray"
    if entry.startswith("dbt "):
        return "dbt"
    if entry.startswith("DVC "):
        return "DVC"
    if entry.startswith("MLflow "):
        return "MLflow"
    match = re.match(r"([A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’`.-]+)", entry)
    return match.group(1) if match else ""


def title_from_entry(entry: str) -> str:
    text = re.sub(r"\*+", "", entry)
    text = re.sub(r"\s+", " ", text).strip()
    year_match = re.search(r"\((?:19|20)\d{2}[a-z]?\)\.?\s*", text)
    if not year_match:
        return ""
    tail = text[year_match.end() :].strip()
    tail = re.sub(r"^[:.]\s*", "", tail)
    tail = re.sub(r"\s+https?://\S+", "", tail)
    tail = re.sub(r"\s+arXiv(?:\s+preprint)?\s+arXiv:\S+\.?$", "", tail, flags=re.I)
    cue_positions = [tail.find(cue) for cue in SOURCE_CUES if tail.find(cue) > 0]
    if cue_positions:
        tail = tail[: min(cue_positions)]
    else:
        sentence = re.split(r"\.\s+", tail, maxsplit=1)
        if sentence:
            tail = sentence[0]
    return tail.strip(" .")


def fingerprint_title(title: str, entry: str) -> str:
    base = title or entry
    return normalize_title(base)[:160]


def parse_reference(path: Path, entry_no: int, line: int, entry: str) -> ReferenceEntry:
    year_match = YEAR_RE.search(entry)
    doi_match = DOI_RE.search(entry)
    arxiv_match = ARXIV_RE.search(entry)
    url_match = URL_RE.search(entry)
    first_author = first_author_from_entry(entry)
    year = parsed_year(year_match)
    title = title_from_entry(entry)
    doi = doi_match.group(0).rstrip(".,;") if doi_match else ""
    arxiv = arxiv_match.group(1).rstrip(".,;") if arxiv_match else ""
    url = clean_url(url_match.group(0)) if url_match else ""
    key = f"{author_key(first_author)}:{year}" if first_author and year else ""
    issues: list[str] = []
    if not year:
        issues.append("missing-year")
    if not first_author:
        issues.append("missing-first-author")
    if not title and not url:
        issues.append("title-not-parsed")
    if url_match and url_match.group(0) != url:
        issues.append("url-trailing-punctuation")
    if not (doi or arxiv or url):
        issues.append("missing-doi-arxiv-url")
    if entry.startswith("[") or "](" in entry:
        issues.append("markdown-link-style")
    if not entry.endswith((".", "。")):
        issues.append("missing-terminal-period")
    return ReferenceEntry(
        file=rel(path),
        line=line,
        entry_no=entry_no,
        entry=entry,
        first_author=first_author,
        year=year,
        title=title,
        doi=doi,
        arxiv=arxiv,
        url=url,
        key=key,
        fingerprint=fingerprint_title(title, entry),
        format_issues=issues,
    )


def should_ignore_citation(author: str, line: str) -> bool:
    word = normalize_word(author)
    if word in NON_REFERENCE_CONTEXT:
        return True
    if word == "standards" and "National Institute of Standards and Technology" in line:
        return True
    if re.match(r"^\s*\|", line):
        # Tables contain legitimate citations too, so do not ignore table rows.
        return False
    return False


def extract_body_citations(path: Path) -> list[BodyCitation]:
    rows: list[BodyCitation] = []
    seen_line_keys: set[tuple[int, str, str]] = set()
    for lineno, line in body_lines_before_references(path):
        for pattern in CITATION_PATTERNS:
            for match in pattern.finditer(line):
                author, year = match.group(1), match.group(2)
                if should_ignore_citation(author, line):
                    continue
                key = f"{author_key(author)}:{year}"
                sig = (lineno, key, match.group(0))
                if sig in seen_line_keys:
                    continue
                seen_line_keys.add(sig)
                context = line.strip()
                if len(context) > 220:
                    context = context[:217] + "..."
                rows.append(
                    BodyCitation(
                        file=rel(path),
                        line=lineno,
                        author=author,
                        year=year,
                        text=match.group(0),
                        key=key,
                        context=context,
                    )
                )
    return rows


def fetch_json(url: str, timeout: float = 8.0) -> dict:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        },
    )
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8", errors="replace"))
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code not in {429, 500, 502, 503, 504}:
                raise
            time.sleep(0.5 * (attempt + 1))
        except Exception as exc:
            last_exc = exc
            time.sleep(0.35 * (attempt + 1))
    if last_exc:
        raise last_exc
    raise RuntimeError("fetch_json failed without exception")


def fetch_text(url: str, timeout: float = 8.0) -> str:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/atom+xml,text/xml,*/*",
        },
    )
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code not in {429, 500, 502, 503, 504}:
                raise
            time.sleep(0.5 * (attempt + 1))
        except Exception as exc:
            last_exc = exc
            time.sleep(0.35 * (attempt + 1))
    if last_exc:
        raise last_exc
    raise RuntimeError("fetch_text failed without exception")


def check_url_status(url: str) -> str:
    if not url:
        return ""
    request = urllib.request.Request(
        url,
        method="HEAD",
        headers={"User-Agent": USER_AGENT},
    )
    try:
        with urllib.request.urlopen(request, timeout=6.0) as response:
            return str(response.status)
    except urllib.error.HTTPError as exc:
        if exc.code in {403, 405}:
            try:
                request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(request, timeout=6.0) as response:
                    return str(response.status)
            except Exception:
                return str(exc.code)
        return str(exc.code)
    except Exception as exc:
        return f"error:{exc.__class__.__name__}"


def verify_arxiv(ref: ReferenceEntry) -> tuple[str, str, str, float, str, list[str]]:
    arxiv_id = re.sub(r"v\d+$", "", ref.arxiv)
    url = "https://export.arxiv.org/api/query?id_list=" + urllib.parse.quote(arxiv_id)
    issues: list[str] = []
    try:
        xml_text = fetch_text(url)
        root = ET.fromstring(xml_text)
        ns = {"a": "http://www.w3.org/2005/Atom"}
        entry = root.find("a:entry", ns)
        if entry is None:
            return "not-found", "arxiv", "", "", 0.0, ref.arxiv, ["arxiv-id-not-found"]
        title = " ".join((entry.findtext("a:title", "", ns) or "").split())
        published = entry.findtext("a:published", "", ns)
        year = published[:4] if published else ""
        score = similarity(ref.title, title) if ref.title else similarity(ref.entry, title)
        if score < 0.62:
            issues.append("title-mismatch")
        if year_delta_too_large(ref.year, year):
            issues.append("year-mismatch")
        status = "verified" if not issues else "metadata-mismatch"
        return status, "arxiv", title, year, score, ref.arxiv, issues
    except Exception as exc:
        return verify_arxiv_html(ref, f"arxiv-error:{exc.__class__.__name__}")


def verify_arxiv_html(ref: ReferenceEntry, prior_issue: str = "") -> tuple[str, str, str, float, str, list[str]]:
    arxiv_id = re.sub(r"v\d+$", "", ref.arxiv)
    url = "https://arxiv.org/abs/" + urllib.parse.quote(arxiv_id)
    issues: list[str] = [prior_issue] if prior_issue else []
    try:
        html_text = fetch_text(url)
        title_match = re.search(r'<meta\s+name="citation_title"\s+content="([^"]+)"', html_text, re.I)
        date_match = re.search(r'<meta\s+name="citation_date"\s+content="([^"]+)"', html_text, re.I)
        title = html.unescape(title_match.group(1)) if title_match else ""
        year = date_match.group(1)[:4] if date_match else ""
        score = similarity(ref.title, title) if ref.title else similarity(ref.entry, title)
        if not title:
            issues.append("arxiv-html-title-not-found")
        elif score < 0.62:
            issues.append("title-mismatch")
        if year_delta_too_large(ref.year, year):
            issues.append("year-mismatch")
        blocking = [issue for issue in issues if not issue.startswith("arxiv-error:")]
        status = "verified" if not blocking else "metadata-mismatch"
        return status, "arxiv-html", title, year, score, ref.arxiv, issues
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return "not-found", "arxiv-html", "", "", 0.0, ref.arxiv, issues + ["arxiv-id-not-found"]
        return "check-error", "arxiv-html", "", "", 0.0, ref.arxiv, issues + [f"arxiv-html-http:{exc.code}"]
    except Exception as exc:
        return "check-error", "arxiv-html", "", "", 0.0, ref.arxiv, issues + [f"arxiv-html-error:{exc.__class__.__name__}"]


def verify_datacite_doi(ref: ReferenceEntry, prior_issue: str = "") -> tuple[str, str, str, float, str, list[str]]:
    doi = ref.doi.lower()
    url = "https://api.datacite.org/dois/" + urllib.parse.quote(doi, safe="")
    issues: list[str] = [prior_issue] if prior_issue else []
    try:
        data = fetch_json(url)
        attrs = data.get("data", {}).get("attributes", {})
        titles = attrs.get("titles") or []
        title = titles[0].get("title", "") if titles else ""
        year = str(attrs.get("publicationYear") or "")
        score = similarity(ref.title, title) if ref.title else similarity(ref.entry, title)
        if score < 0.62:
            issues.append("title-mismatch")
        if year_delta_too_large(ref.year, year):
            issues.append("year-mismatch")
        blocking = [issue for issue in issues if issue != "doi-not-found-crossref"]
        status = "verified" if not blocking else "metadata-mismatch"
        return status, "datacite-doi", title, year, score, doi, issues
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return "not-found", "datacite-doi", "", "", 0.0, doi, issues + ["doi-not-found-datacite"]
        return "check-error", "datacite-doi", "", "", 0.0, doi, issues + [f"datacite-http:{exc.code}"]
    except Exception as exc:
        return "check-error", "datacite-doi", "", "", 0.0, doi, issues + [f"datacite-error:{exc.__class__.__name__}"]


def verify_doi(ref: ReferenceEntry) -> tuple[str, str, str, float, str, list[str]]:
    doi = ref.doi.lower()
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="")
    issues: list[str] = []
    try:
        data = fetch_json(url)
        message = data.get("message", {})
        titles = message.get("title") or []
        title = titles[0] if titles else ""
        issued = message.get("issued", {}).get("date-parts", [])
        year = str(issued[0][0]) if issued and issued[0] and issued[0][0] else ""
        score = similarity(ref.title, title) if ref.title else similarity(ref.entry, title)
        if score < 0.62:
            issues.append("title-mismatch")
        if year_delta_too_large(ref.year, year):
            issues.append("year-mismatch")
        status = "verified" if not issues else "metadata-mismatch"
        return status, "crossref-doi", title, year, score, doi, issues
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return verify_datacite_doi(ref, "doi-not-found-crossref")
        return "check-error", "crossref-doi", "", "", 0.0, doi, [f"doi-http:{exc.code}"]
    except Exception as exc:
        datacite_status, datacite_source, datacite_title, datacite_year, datacite_score, datacite_id, datacite_issues = verify_datacite_doi(
            ref, f"doi-error:{exc.__class__.__name__}"
        )
        if datacite_status != "check-error":
            return datacite_status, datacite_source, datacite_title, datacite_year, datacite_score, datacite_id, datacite_issues
        return "check-error", "crossref-doi", "", "", 0.0, doi, [f"doi-error:{exc.__class__.__name__}"]


def crossref_search(ref: ReferenceEntry) -> tuple[str, str, str, float, str, list[str]]:
    query = ref.title or ref.entry
    params = urllib.parse.urlencode(
        {
            "query.title": query,
            "rows": "3",
            "select": "DOI,title,issued,published-print,published-online,type,URL",
        }
    )
    url = "https://api.crossref.org/works?" + params
    try:
        data = fetch_json(url)
        items = data.get("message", {}).get("items", [])
    except Exception as exc:
        return "check-error", "crossref-search", "", "", 0.0, "", [f"crossref-error:{exc.__class__.__name__}"]
    best: tuple[float, dict] = (0.0, {})
    for item in items:
        title = (item.get("title") or [""])[0]
        score = similarity(ref.title or ref.entry, title)
        if score > best[0]:
            best = (score, item)
    if not best[1]:
        return "not-found", "crossref-search", "", "", 0.0, "", ["crossref-no-result"]
    item = best[1]
    title = (item.get("title") or [""])[0]
    issued = item.get("issued", {}).get("date-parts", [])
    year = str(issued[0][0]) if issued and issued[0] and issued[0][0] else ""
    identifier = item.get("DOI") or item.get("URL") or ""
    issues: list[str] = []
    if best[0] < 0.72:
        issues.append("weak-title-match")
    if year_delta_too_large(ref.year, year):
        issues.append("year-mismatch")
    status = "verified" if not issues else "metadata-mismatch"
    return status, "crossref-search", title, year, best[0], identifier, issues


def openalex_search(ref: ReferenceEntry) -> tuple[str, str, str, float, str, list[str]]:
    query = ref.title or ref.entry
    params = urllib.parse.urlencode({"search": query, "per-page": "3"})
    url = "https://api.openalex.org/works?" + params
    try:
        data = fetch_json(url)
        items = data.get("results", [])
    except Exception as exc:
        return "check-error", "openalex-search", "", "", 0.0, "", [f"openalex-error:{exc.__class__.__name__}"]
    best: tuple[float, dict] = (0.0, {})
    for item in items:
        title = item.get("title") or ""
        score = similarity(ref.title or ref.entry, title)
        if score > best[0]:
            best = (score, item)
    if not best[1]:
        return "not-found", "openalex-search", "", "", 0.0, "", ["openalex-no-result"]
    item = best[1]
    title = item.get("title") or ""
    year = str(item.get("publication_year") or "")
    identifier = item.get("doi") or item.get("id") or ""
    issues: list[str] = []
    if best[0] < 0.72:
        issues.append("weak-title-match")
    if year_delta_too_large(ref.year, year):
        issues.append("year-mismatch")
    status = "verified" if not issues else "metadata-mismatch"
    return status, "openalex-search", title, year, best[0], identifier, issues


def external_check(ref: ReferenceEntry, sleep_s: float, title_search: bool) -> ExternalCheck:
    url_status = check_url_status(ref.url) if ref.url else ""
    if ref.arxiv:
        status, source, title, year, score, identifier, issues = verify_arxiv(ref)
    elif ref.doi:
        status, source, title, year, score, identifier, issues = verify_doi(ref)
    elif title_search and ref.title and len(normalize_title(ref.title)) >= 16 and not ref.url:
        status, source, title, year, score, identifier, issues = crossref_search(ref)
        if status in {"not-found", "metadata-mismatch", "check-error"}:
            alt_status, alt_source, alt_title, alt_year, alt_score, alt_identifier, alt_issues = openalex_search(ref)
            if alt_status == "verified" or (alt_score > score and alt_status != "check-error"):
                status, source, title, year, score, identifier, issues = (
                    alt_status,
                    alt_source,
                    alt_title,
                    alt_year,
                    alt_score,
                    alt_identifier,
                    alt_issues,
                )
    elif ref.url:
        ok = url_status.startswith(("2", "3")) or url_status == "403"
        status = "url-reachable" if ok else "url-problem"
        source = "url"
        title = ""
        year = ""
        score = 0.0
        identifier = ref.url
        issues = [] if ok else [f"url-status:{url_status}"]
    else:
        status = "needs-manual-review"
        source = "none"
        title = ""
        year = ""
        score = 0.0
        identifier = ""
        issues = ["no-machine-verifiable-identifier"]
    if sleep_s:
        time.sleep(sleep_s)
    return ExternalCheck(
        file=ref.file,
        line=ref.line,
        entry_no=ref.entry_no,
        key=ref.key,
        title=ref.title,
        status=status,
        source=source,
        matched_title=title,
        matched_year=year,
        score=round(score, 3),
        identifier=identifier,
        url_status=url_status,
        issues=issues,
    )


def external_cache_key(ref: ReferenceEntry) -> str:
    if ref.arxiv:
        return "arxiv:" + re.sub(r"v\d+$", "", ref.arxiv.lower())
    if ref.doi:
        return "doi:" + ref.doi.lower()
    if ref.url and not ref.title:
        return "url:" + ref.url
    return "title:" + (ref.fingerprint or normalize_title(ref.entry))


def clone_check(template: ExternalCheck, ref: ReferenceEntry) -> ExternalCheck:
    return ExternalCheck(
        file=ref.file,
        line=ref.line,
        entry_no=ref.entry_no,
        key=ref.key,
        title=ref.title,
        status=template.status,
        source=template.source,
        matched_title=template.matched_title,
        matched_year=template.matched_year,
        score=template.score,
        identifier=template.identifier,
        url_status=template.url_status,
        issues=list(template.issues),
    )


def run_external_checks(references: list[ReferenceEntry], sleep_s: float, workers: int, title_search: bool) -> list[ExternalCheck]:
    grouped: dict[str, list[ReferenceEntry]] = defaultdict(list)
    for ref in references:
        grouped[external_cache_key(ref)].append(ref)

    templates: dict[str, ExternalCheck] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        future_map = {
            executor.submit(external_check, refs[0], sleep_s, title_search): key
            for key, refs in grouped.items()
        }
        for future in as_completed(future_map):
            key = future_map[future]
            try:
                templates[key] = future.result()
            except Exception as exc:
                ref = grouped[key][0]
                templates[key] = ExternalCheck(
                    file=ref.file,
                    line=ref.line,
                    entry_no=ref.entry_no,
                    key=ref.key,
                    title=ref.title,
                    status="check-error",
                    source="script",
                    matched_title="",
                    matched_year="",
                    score=0.0,
                    identifier="",
                    url_status="",
                    issues=[f"unhandled-error:{exc.__class__.__name__}"],
                )

    return [clone_check(templates[external_cache_key(ref)], ref) for ref in references]


def write_markdown(
    path: Path,
    files: list[Path],
    references: list[ReferenceEntry],
    citations: list[BodyCitation],
    checks: list[ExternalCheck],
    missing_citations: list[BodyCitation],
    uncited_refs: list[ReferenceEntry],
    duplicate_refs: dict[tuple[str, str], list[ReferenceEntry]],
    max_rows: int,
) -> None:
    status_counts = Counter(row.status for row in checks)
    format_issue_counts = Counter(issue for ref in references for issue in ref.format_issues)
    out: list[str] = [
        "# 全书引用完整性与真实性审计报告",
        "",
        f"- 生成时间：{datetime.now(timezone.utc).isoformat()}",
        "- 范围：当前 Springer 中文交付稿 `docs/zh` 中的正文章、项目章和附录。",
        f"- 扫描文件：{len(files)}",
        f"- 参考文献条目：{len(references)}",
        f"- 正文 author-year 引用：{len(citations)}",
        f"- 正文引用未在同章参考文献解析到：{len(missing_citations)}",
        f"- 章末参考文献未被同章正文引用：{len(uncited_refs)}",
        f"- 同章疑似重复参考文献组：{len(duplicate_refs)}",
        "",
        "## 外部核验概览",
        "",
        "| 状态 | 数量 | 含义 |",
        "| --- | ---: | --- |",
    ]
    status_labels = {
        "verified": "arXiv / DOI / DataCite / Crossref / OpenAlex 强匹配",
        "metadata-mismatch": "外部记录存在，但题名或年份弱匹配/不一致",
        "url-reachable": "URL 可访问，但非论文元数据强校验",
        "url-problem": "URL 不可访问或状态异常",
        "not-found": "外部元数据未查到",
        "check-error": "外部服务请求失败",
        "needs-manual-review": "缺少可机器核验标识",
    }
    for status, count in sorted(status_counts.items()):
        out.append(f"| `{status}` | {count} | {status_labels.get(status, '')} |")
    out.extend(["", "## 主要结论", ""])
    blockers = [row for row in checks if row.status in {"metadata-mismatch", "url-problem", "not-found", "needs-manual-review"}]
    no_identifier = format_issue_counts.get("missing-doi-arxiv-url", 0)
    out.append(f"- 需要优先人工复核的外部核验问题：{len(blockers)} 条。")
    out.append(f"- 缺少 DOI / arXiv / URL 的条目：{no_identifier} 条；其中一部分可由 Crossref/OpenAlex 题名检索确认，但 Springer 终稿仍建议补 DOI 或稳定 URL。")
    out.append("- `url-reachable` 只能证明网页当前可达，不能证明引文格式、版本日期和题名完全符合出版社要求。")
    out.append("- 当前报告只确认“同章 author-year 可解析对应关系”；对于一段话是否应该引用更精确来源，仍需人工学术编辑判断。")
    out.extend(["", "## 正文引用未解析到同章参考文献", ""])
    out.extend(["| 文件 | 行 | 引用 | 上下文 |", "| --- | ---: | --- | --- |"])
    for item in missing_citations[:max_rows]:
        context = item.context.replace("|", "\\|")
        out.append(f"| `{item.file}` | {item.line} | `{item.text}` | {context} |")
    if len(missing_citations) > max_rows:
        out.append(f"\n> 其余 {len(missing_citations) - max_rows} 条见 JSON 明细。")
    out.extend(["", "## 章末参考文献未被正文引用", ""])
    out.extend(["| 文件 | 行 | 序号 | Key | 题名 | 条目 |", "| --- | ---: | ---: | --- | --- | --- |"])
    for ref in uncited_refs[:max_rows]:
        entry = ref.entry.replace("|", "\\|")
        title = ref.title.replace("|", "\\|")
        if len(entry) > 220:
            entry = entry[:217] + "..."
        out.append(f"| `{ref.file}` | {ref.line} | {ref.entry_no} | `{ref.key}` | {title} | {entry} |")
    if len(uncited_refs) > max_rows:
        out.append(f"\n> 其余 {len(uncited_refs) - max_rows} 条见 JSON 明细。")
    out.extend(["", "## 外部核验问题条目", ""])
    out.extend(["| 文件 | 行 | 序号 | 状态 | 来源 | 分数 | 标识 | 问题 | 题名 | 匹配题名 |", "| --- | ---: | ---: | --- | --- | ---: | --- | --- | --- | --- |"])
    for row in blockers[:max_rows]:
        title = row.title.replace("|", "\\|")
        matched = row.matched_title.replace("|", "\\|")
        if len(title) > 120:
            title = title[:117] + "..."
        if len(matched) > 120:
            matched = matched[:117] + "..."
        identifier = row.identifier.replace("|", "\\|")
        issues = ", ".join(row.issues).replace("|", "\\|")
        out.append(f"| `{row.file}` | {row.line} | {row.entry_no} | `{row.status}` | {row.source} | {row.score:.3f} | {identifier} | {issues} | {title} | {matched} |")
    if len(blockers) > max_rows:
        out.append(f"\n> 其余 {len(blockers) - max_rows} 条见 JSON 明细。")
    out.extend(["", "## 格式问题统计", ""])
    out.extend(["| 问题 | 数量 |", "| --- | ---: |"])
    for issue, count in sorted(format_issue_counts.items()):
        out.append(f"| `{issue}` | {count} |")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit reference integrity for Springer manuscript.")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--roots", nargs="*", type=Path, default=DEFAULT_ROOTS)
    parser.add_argument("--sleep", type=float, default=0.1, help="Delay inside each external worker after a check.")
    parser.add_argument("--workers", type=int, default=6, help="Concurrent external metadata checks.")
    parser.add_argument(
        "--title-search",
        action="store_true",
        help="Also query Crossref/OpenAlex by title for entries without DOI/arXiv/URL. Slower and may produce weak matches.",
    )
    parser.add_argument("--max-rows", type=int, default=120)
    parser.add_argument("--skip-external", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    roots = [(ROOT / path).resolve() if not path.is_absolute() else path.resolve() for path in args.roots]
    files = book_files(roots)
    references: list[ReferenceEntry] = []
    citations: list[BodyCitation] = []
    for path in files:
        for idx, (line, entry) in enumerate(extract_reference_section(path), 1):
            references.append(parse_reference(path, idx, line, entry))
        citations.extend(extract_body_citations(path))

    refs_by_file_key: dict[tuple[str, str], list[ReferenceEntry]] = defaultdict(list)
    for ref in references:
        if ref.key:
            refs_by_file_key[(ref.file, ref.key)].append(ref)

    missing_citations = [citation for citation in citations if (citation.file, citation.key) not in refs_by_file_key]

    cited_keys_by_file: dict[str, set[str]] = defaultdict(set)
    for citation in citations:
        cited_keys_by_file[citation.file].add(citation.key)
    uncited_refs = [ref for ref in references if ref.key and ref.key not in cited_keys_by_file.get(ref.file, set())]

    duplicate_refs: dict[tuple[str, str], list[ReferenceEntry]] = defaultdict(list)
    for ref in references:
        if ref.fingerprint:
            duplicate_refs[(ref.file, ref.fingerprint)].append(ref)
    duplicate_refs = {key: value for key, value in duplicate_refs.items() if len(value) > 1}

    if args.skip_external:
        checks = [
            ExternalCheck(ref.file, ref.line, ref.entry_no, ref.key, ref.title, "not-checked", "", "", "", 0.0, "", "", [])
            for ref in references
        ]
    else:
        checks = run_external_checks(references, args.sleep, args.workers, args.title_search)

    report_dir = args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope_roots": [rel(path) if path.is_relative_to(ROOT) else str(path) for path in roots],
        "summary": {
            "files": len(files),
            "references": len(references),
            "body_author_year_citations": len(citations),
            "missing_same_chapter_references": len(missing_citations),
            "uncited_references": len(uncited_refs),
            "duplicate_reference_groups": len(duplicate_refs),
            "external_status_counts": dict(Counter(row.status for row in checks)),
            "format_issue_counts": dict(Counter(issue for ref in references for issue in ref.format_issues)),
        },
        "references": [asdict(ref) for ref in references],
        "body_citations": [asdict(citation) for citation in citations],
        "missing_same_chapter_references": [asdict(citation) for citation in missing_citations],
        "uncited_references": [asdict(ref) for ref in uncited_refs],
        "duplicate_reference_groups": {
            f"{file}::{fingerprint}": [asdict(ref) for ref in refs] for (file, fingerprint), refs in duplicate_refs.items()
        },
        "external_checks": [asdict(check) for check in checks],
    }
    (report_dir / "reference_integrity_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown(
        report_dir / "reference_integrity_audit.md",
        files,
        references,
        citations,
        checks,
        missing_citations,
        uncited_refs,
        duplicate_refs,
        args.max_rows,
    )
    print("Reference integrity audit generated")
    for key, value in payload["summary"].items():
        print(f"- {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
