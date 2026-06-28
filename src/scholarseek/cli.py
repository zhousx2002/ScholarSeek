from __future__ import annotations

import argparse
import os
import sys

from .config import get_api_config, mask_secret
from .formatters import format_json, format_markdown
from .query_planner import build_query_plan
from .qwen import DEFAULT_QWEN_BASE_URL, DEFAULT_QWEN_MODEL, build_qwen_query_plan, synthesize_qwen_answer
from .ranker import rank_papers
from .retrievers import MultiSourceRetriever, parse_sources


def main(argv: list[str] | None = None) -> int:
    config = get_api_config()
    parser = argparse.ArgumentParser(description="Search academic papers for a complex scholarly query.")
    parser.add_argument("query", help="Natural language scholarly query.")
    parser.add_argument("--max-queries", type=int, default=6, help="Maximum generated search queries.")
    parser.add_argument("--per-query", type=int, default=10, help="OpenAlex results per search query.")
    parser.add_argument("--limit", type=int, default=15, help="Number of ranked papers to output.")
    parser.add_argument("--email", default=config.openalex_email, help="Email for OpenAlex polite pool.")
    parser.add_argument("--openalex-api-key", default=config.openalex_api_key)
    parser.add_argument(
        "--sources",
        default=config.sources,
        help="Comma-separated retrieval sources: openalex, semantic-scholar, arxiv, or all.",
    )
    parser.add_argument("--semantic-scholar-api-key", default=config.semantic_scholar_api_key)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--planner", choices=("heuristic", "qwen"), default="heuristic")
    parser.add_argument("--answer", choices=("none", "qwen"), default="none")
    parser.add_argument("--strategy", choices=("standard", "spar", "spar-qwen"), default="standard")
    parser.add_argument(
        "--qwen-base-url",
        default=config.qwen_base_url,
    )
    parser.add_argument("--qwen-model", default=config.qwen_model)
    parser.add_argument("--qwen-api-key", default=config.qwen_api_key)
    parser.add_argument("--show-api-config", action="store_true", help="Print API key configuration status and exit.")
    parser.add_argument(
        "--strict-qwen",
        action="store_true",
        help="Fail instead of falling back to heuristic planning when Qwen is unavailable.",
    )
    args = parser.parse_args(argv)

    if args.show_api_config:
        print(f"Qwen base URL: {args.qwen_base_url}")
        print(f"Qwen model: {args.qwen_model}")
        print(f"Qwen API key: {mask_secret(args.qwen_api_key)}")
        print(f"Semantic Scholar API key: {mask_secret(args.semantic_scholar_api_key)}")
        print(f"OpenAlex email: {args.email or 'missing'}")
        print(f"OpenAlex API key: {mask_secret(args.openalex_api_key)}")
        print(f"Sources: {args.sources}")
        print(f"Trainable reranker path: {config.reranker_path or 'missing'}")
        return 0

    if args.planner == "qwen":
        plan = build_qwen_query_plan(
            args.query,
            max_queries=args.max_queries,
            base_url=args.qwen_base_url,
            model=args.qwen_model,
            api_key=args.qwen_api_key,
            fallback_on_error=not args.strict_qwen,
        )
    else:
        plan = build_query_plan(args.query, max_queries=args.max_queries)
    selected_sources = parse_sources(args.sources)
    retriever = MultiSourceRetriever(
        sources=selected_sources,
        openalex_email=args.email,
        openalex_api_key=args.openalex_api_key,
        semantic_scholar_api_key=args.semantic_scholar_api_key,
    )
    papers = retriever.search(plan, per_query=args.per_query)
    if args.strategy in {"spar", "spar-qwen"}:
        from .citation_expander import CitationExpander
        from .judgement import JudgementAgent
        from .refchain_planner import RefChainPlanner

        enable_openalex_refchain = _env_bool("SCHOLARSEEK_ENABLE_OPENALEX_REFCHAIN", False)
        enable_citation_expansion = ("semantic-scholar" in selected_sources) or (
            enable_openalex_refchain and "openalex" in selected_sources
        )
        spar_result = RefChainPlanner(
            retriever=retriever,
            judgement_agent=JudgementAgent(
                reranker_path=config.reranker_path,
                use_qwen=args.strategy == "spar-qwen",
                qwen_base_url=args.qwen_base_url,
                qwen_model=args.qwen_model,
                qwen_api_key=args.qwen_api_key,
            ),
            citation_expander=CitationExpander(
                api_key=args.semantic_scholar_api_key,
                enabled=enable_citation_expansion,
                openalex_email=args.email,
                openalex_api_key=args.openalex_api_key,
                use_openalex=enable_openalex_refchain and "openalex" in selected_sources,
            ),
            per_query=args.per_query,
        ).run(
            plan,
            papers,
            args.limit,
            qwen_config={
                "base_url": args.qwen_base_url,
                "model": args.qwen_model,
                "api_key": args.qwen_api_key if args.strategy == "spar-qwen" else None,
            },
        )
        ranked = spar_result.papers
    else:
        ranked = rank_papers(papers, plan, limit=args.limit)
    answer = None
    if args.answer == "qwen":
        answer = synthesize_qwen_answer(
            args.query,
            ranked,
            base_url=args.qwen_base_url,
            model=args.qwen_model,
            api_key=args.qwen_api_key,
            fallback_on_error=not args.strict_qwen,
        )

    if args.format == "json":
        print(format_json(plan, ranked, answer=answer))
    else:
        print(format_markdown(plan, ranked, answer=answer))
    return 0


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


if __name__ == "__main__":
    sys.exit(main())
