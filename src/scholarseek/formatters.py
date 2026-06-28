from __future__ import annotations

import json
from typing import Iterable

from .models import Paper, QueryPlan


def format_markdown(plan: QueryPlan, papers: Iterable[Paper], answer: str | None = None) -> str:
    lines = [
        "# ScholarSeek Results",
        "",
        f"**Original query:** {plan.original}",
        f"**Planner:** {plan.planner}",
        "",
        "## Query Plan",
        "",
        "- Search queries: " + "; ".join(plan.search_queries),
        "- Core terms: " + (", ".join(plan.must_terms) or "none"),
        "- Expanded terms: " + (", ".join(plan.optional_terms) or "none"),
    ]
    if plan.year_from or plan.year_to:
        lines.append(f"- Year range: {plan.year_from or '*'} - {plan.year_to or '*'}")

    if answer:
        lines.extend(["", "## Answer", "", answer])

    lines.extend(["", "## Ranked Papers", ""])
    for index, paper in enumerate(papers, start=1):
        authors = ", ".join(paper.authors[:4])
        if len(paper.authors) > 4:
            authors += ", et al."
        meta = " | ".join(
            item
            for item in [
                str(paper.year) if paper.year else "",
                paper.venue,
                f"{paper.citation_count} citations",
                paper.source,
            ]
            if item
        )
        lines.extend(
            [
                f"### {index}. {paper.title}",
                "",
                f"- Score: {paper.score:.3f}",
                f"- Metadata: {meta}",
                f"- Authors: {authors or 'unknown'}",
                f"- Reason: {paper.relevance_reason}",
                f"- URL: {paper.url or paper.doi or paper.id}",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def format_json(plan: QueryPlan, papers: Iterable[Paper], answer: str | None = None) -> str:
    payload = {
        "answer": answer,
        "query_plan": {
            "original": plan.original,
            "search_queries": plan.search_queries,
            "must_terms": plan.must_terms,
            "optional_terms": plan.optional_terms,
            "year_from": plan.year_from,
            "year_to": plan.year_to,
            "planner": plan.planner,
        },
        "papers": [
            {
                "rank": rank,
                "id": paper.id,
                "score": paper.score,
                "title": paper.title,
                "year": paper.year,
                "venue": paper.venue,
                "authors": paper.authors,
                "abstract": paper.abstract,
                "doi": paper.doi,
                "url": paper.url,
                "citation_count": paper.citation_count,
                "source": paper.source,
                "reason": paper.relevance_reason,
                "raw": paper.raw,
            }
            for rank, paper in enumerate(papers, start=1)
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
