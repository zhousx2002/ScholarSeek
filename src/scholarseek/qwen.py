from __future__ import annotations

import json
import os
import re
import socket
from dataclasses import replace
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from .models import Paper, QueryPlan
from .query_planner import build_query_plan


DEFAULT_QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_QWEN_MODEL = "qwen-plus"
DEFAULT_QWEN_PLANNER_TIMEOUT = 35
DEFAULT_QWEN_ANSWER_TIMEOUT = 45
DEFAULT_QWEN_JSON_TIMEOUT = 30


class QwenPlanner:
    """Query planner using a Qwen OpenAI-compatible chat completion endpoint."""

    def __init__(
        self,
        base_url: str = DEFAULT_QWEN_BASE_URL,
        model: str = DEFAULT_QWEN_MODEL,
        api_key: Optional[str] = None,
        timeout: int = DEFAULT_QWEN_PLANNER_TIMEOUT,
    ):
        self.base_url = base_url.rstrip("/") + "/"
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def build_plan(self, query: str, max_queries: int = 6) -> QueryPlan:
        fallback = build_query_plan(query, max_queries=max_queries)
        payload = self._chat_payload(query, fallback, max_queries)
        data = self._post_chat(payload)
        content = _message_content(data)
        parsed = _extract_json_object(content)
        if not parsed:
            return fallback
        return _merge_plan(query, parsed, fallback, max_queries, self.model)

    def _chat_payload(self, query: str, fallback: QueryPlan, max_queries: int) -> Dict[str, Any]:
        system = (
            "You are ScholarSeek's academic search query planner. "
            "Return only valid JSON. No markdown. "
            "Extract research intent, constraints, and English search queries for academic APIs. "
            "For benchmark-style questions that describe a known paper indirectly, infer likely canonical "
            "paper titles, method names, dataset names, or acronyms and put them in candidate_paper_titles. "
            "candidate_paper_titles must be specific paper-like titles or distinctive title phrases, not broad "
            "research areas. For example, prefer 'BEVFormer' or 'Encode, Tag, Realize' over "
            "'spatiotemporal transformer' or 'text editing'. "
            "If you are uncertain, still provide several plausible exact titles, acronyms, or unique method names. "
            "Create a diversified retrieval plan rather than paraphrases of the same query. "
            "For broad natural-language requests, include distinct query types: "
            "1) core task and method keywords, "
            "2) likely paper-title phrases or named methods, "
            "3) benchmark/dataset/application terms, "
            "4) survey/review query when the user asks for overviews, "
            "5) concise field-specific terminology used by authors. "
            "Prefer short keyword queries of 3 to 9 words; avoid full-sentence questions."
        )
        user = {
            "query": query,
            "max_search_queries": max_queries,
            "fallback_terms": {
                "must_terms": fallback.must_terms,
                "optional_terms": fallback.optional_terms,
                "year_from": fallback.year_from,
                "year_to": fallback.year_to,
            },
            "planner_priority": [
                "First infer exact paper titles, method names, acronyms, and distinctive title phrases.",
                "Then produce compact keyword search queries for academic APIs.",
                "Do not let generic fallback terms dominate the search plan.",
            ],
            "required_schema": {
                "search_queries": ["string"],
                "candidate_paper_titles": ["specific paper title, method acronym, or distinctive title phrase"],
                "must_terms": ["string"],
                "optional_terms": ["string"],
                "year_from": "integer or null",
                "year_to": "integer or null",
            },
            "search_query_rules": [
                "Use English only.",
                "Do not include Boolean operators.",
                "Do not repeat near-duplicate queries.",
                "When a likely paper title or named method is known, include it as a short exact-title query.",
                "Include acronyms with expansions when useful, e.g. GAN generative adversarial network.",
                "If the request mentions applications, include one method query and one application query.",
                "If the request asks for latest/recent, keep the year fields but avoid putting years into every query.",
            ],
        }
        return {
            "model": self.model,
            "temperature": 0.0,
            "max_tokens": 700,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
            ],
        }

    def _post_chat(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = urljoin(self.base_url, "chat/completions")
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(url, data=body, headers=headers, method="POST")
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, socket.timeout, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Qwen planner request failed: {exc}") from exc


class QwenAnswerSynthesizer:
    """Generate a structured answer from ranked paper evidence using Qwen."""

    def __init__(
        self,
        base_url: str = DEFAULT_QWEN_BASE_URL,
        model: str = DEFAULT_QWEN_MODEL,
        api_key: Optional[str] = None,
        timeout: int = DEFAULT_QWEN_ANSWER_TIMEOUT,
    ):
        self.base_url = base_url.rstrip("/") + "/"
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def synthesize(self, query: str, papers: List[Paper], max_papers: int = 8) -> str:
        evidence = [_paper_evidence(paper, index) for index, paper in enumerate(papers[:max_papers], start=1)]
        payload = {
            "model": self.model,
            "temperature": 0.2,
            "max_tokens": 900,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are ScholarSeek's scientific literature assistant. "
                        "Answer the user using only the provided paper evidence. "
                        "Write in Chinese when the user query is Chinese; otherwise write in English. "
                        "Be concise, structured, and cite papers as [1], [2]. "
                        "If evidence is weak, say which parts need verification."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "query": query,
                            "papers": evidence,
                            "expected_sections": [
                                "direct_answer",
                                "highly_relevant_papers",
                                "partial_or_needs_verification",
                                "search_gaps",
                            ],
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
        }
        data = self._post_chat(payload)
        return _message_content(data).strip()

    def _post_chat(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = urljoin(self.base_url, "chat/completions")
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(url, data=body, headers=headers, method="POST")
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, socket.timeout, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Qwen answer request failed: {exc}") from exc


def build_qwen_query_plan(
    query: str,
    max_queries: int = 6,
    base_url: str = DEFAULT_QWEN_BASE_URL,
    model: str = DEFAULT_QWEN_MODEL,
    api_key: Optional[str] = None,
    fallback_on_error: bool = True,
) -> QueryPlan:
    planner = QwenPlanner(
        base_url=base_url,
        model=model,
        api_key=api_key,
        timeout=_env_int("QWEN_PLANNER_TIMEOUT", DEFAULT_QWEN_PLANNER_TIMEOUT),
    )
    try:
        return planner.build_plan(query, max_queries=max_queries)
    except RuntimeError:
        if not fallback_on_error:
            raise
        fallback = build_query_plan(query, max_queries=max_queries)
        return replace(fallback, planner=f"heuristic-fallback:{model}")


def synthesize_qwen_answer(
    query: str,
    papers: List[Paper],
    base_url: str = DEFAULT_QWEN_BASE_URL,
    model: str = DEFAULT_QWEN_MODEL,
    api_key: Optional[str] = None,
    fallback_on_error: bool = True,
) -> str:
    synthesizer = QwenAnswerSynthesizer(
        base_url=base_url,
        model=model,
        api_key=api_key,
        timeout=_env_int("QWEN_ANSWER_TIMEOUT", DEFAULT_QWEN_ANSWER_TIMEOUT),
    )
    try:
        return synthesizer.synthesize(query, papers)
    except RuntimeError:
        if not fallback_on_error:
            raise
        return _fallback_answer(query, papers)


def request_qwen_json(
    system_prompt: str,
    payload: Dict[str, Any],
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.1,
    timeout: Optional[int] = None,
    max_tokens: int = 700,
) -> Dict[str, Any]:
    client = QwenPlanner(
        base_url=base_url or DEFAULT_QWEN_BASE_URL,
        model=model or DEFAULT_QWEN_MODEL,
        api_key=api_key,
        timeout=timeout or _env_int("QWEN_JSON_TIMEOUT", DEFAULT_QWEN_JSON_TIMEOUT),
    )
    response = client._post_chat(
        {
            "model": client.model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        }
    )
    parsed = _extract_json_object(_message_content(response))
    if parsed is None:
        raise RuntimeError("Qwen returned invalid JSON.")
    return parsed


def _message_content(data: Dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content") or ""
    if isinstance(content, list):
        return "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return str(content)


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.I).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    match = re.search(r"\{.*\}", stripped, flags=re.S)
    candidate = match.group(0) if match else stripped
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _merge_plan(
    query: str,
    parsed: Dict[str, Any],
    fallback: QueryPlan,
    max_queries: int,
    model: str,
) -> QueryPlan:
    generated_queries = _clean_list(parsed.get("search_queries"))
    title_hints = _clean_title_hints(parsed.get("candidate_paper_titles"))
    search_queries = _select_search_queries(
        title_hints=title_hints,
        generated_queries=generated_queries,
        fallback_queries=fallback.search_queries,
        original=query,
        max_queries=max_queries,
    )
    generated_must_terms = _clean_list(parsed.get("must_terms"))
    generated_optional_terms = _clean_list(parsed.get("optional_terms"))
    must_terms = _clean_search_queries([*generated_must_terms, *fallback.must_terms])
    optional_terms = _clean_search_queries([*title_hints, *generated_optional_terms, *fallback.optional_terms])
    year_from = _clean_year(parsed.get("year_from"), fallback.year_from)
    year_to = _clean_year(parsed.get("year_to"), fallback.year_to)

    if not search_queries:
        search_queries = fallback.search_queries
    return QueryPlan(
        original=query,
        search_queries=search_queries,
        must_terms=must_terms,
        optional_terms=optional_terms or fallback.optional_terms,
        year_from=year_from,
        year_to=year_to,
        planner=f"qwen:{model}",
    )


def _select_search_queries(
    *,
    title_hints: List[str],
    generated_queries: List[str],
    fallback_queries: List[str],
    original: str,
    max_queries: int,
) -> List[str]:
    selected: List[str] = []
    pools = [
        title_hints[:2],
        generated_queries[:2],
        fallback_queries[:1],
        generated_queries[2:],
        fallback_queries[1:],
        [original],
    ]
    for pool in pools:
        for query in _clean_search_queries(pool):
            key = query.lower()
            if key not in {item.lower() for item in selected}:
                selected.append(query)
            if len(selected) >= max_queries:
                return selected
    return selected


def _clean_title_hints(value: Any) -> List[str]:
    hints = []
    for item in _clean_list(value):
        text = re.sub(r"\s+", " ", item).strip(" .;:")
        words = text.split()
        if 2 <= len(words) <= 16:
            hints.append(text)
    return _clean_search_queries(hints)


def _clean_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    cleaned = []
    seen = set()
    for item in value:
        text = " ".join(str(item).strip().split())
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            cleaned.append(text)
    return cleaned


def _clean_search_queries(value: List[str]) -> List[str]:
    cleaned = []
    seen = set()
    for item in value:
        text = re.sub(r"\bAND\b|\bOR\b", " ", str(item), flags=re.I)
        text = text.replace('"', " ").replace("'", " ")
        text = " ".join(text.split())
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            cleaned.append(text)
    return cleaned


def _clean_year(value: Any, fallback: Optional[int]) -> Optional[int]:
    if value is None:
        return fallback
    try:
        year = int(value)
    except (TypeError, ValueError):
        return fallback
    if 1900 <= year <= 2100:
        return year
    return fallback


def _paper_evidence(paper: Paper, index: int) -> Dict[str, Any]:
    abstract = paper.abstract[:700] if paper.abstract else ""
    return {
        "index": index,
        "title": paper.title,
        "year": paper.year,
        "venue": paper.venue,
        "authors": paper.authors[:6],
        "score": round(paper.score, 4),
        "citation_count": paper.citation_count,
        "reason": paper.relevance_reason,
        "abstract": abstract,
        "url": paper.url or paper.doi or paper.id,
    }


def _fallback_answer(query: str, papers: List[Paper]) -> str:
    if not papers:
        return "No candidate papers were retrieved, so ScholarSeek cannot synthesize an evidence-based answer."
    top = papers[:5]
    lines = [
        "Qwen answer generation was unavailable, so this fallback summarizes the ranked evidence.",
        "",
        f"Query: {query}",
        "",
        "Most relevant candidates:",
    ]
    for index, paper in enumerate(top, start=1):
        year = paper.year or "unknown year"
        lines.append(f"{index}. {paper.title} ({year}) - {paper.relevance_reason}")
    return "\n".join(lines)


def _env_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, "").strip())
    except ValueError:
        return default
    return value if value > 0 else default
