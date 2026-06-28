from __future__ import annotations

import argparse
import json
import mimetypes
import os
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import unquote, urlparse

from .config import get_api_config, mask_secret
from .search_service import search_papers, synthesize_answer_for_papers


ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = ROOT / "frontend"


class ScholarSeekHandler(BaseHTTPRequestHandler):
    server_version = "ScholarSeekHTTP/0.1"

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors_headers()
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            self._json({"ok": True})
            return
        if parsed.path == "/api/config":
            config = get_api_config()
            self._json(
                {
                    "qwen_base_url": config.qwen_base_url,
                    "qwen_model": config.qwen_model,
                    "qwen_api_key": mask_secret(config.qwen_api_key),
                    "semantic_scholar_api_key": mask_secret(config.semantic_scholar_api_key),
                    "openalex_email": config.openalex_email or "missing",
                    "openalex_api_key": mask_secret(config.openalex_api_key),
                    "sources": config.sources,
                    "local_dataset_dir": os.getenv("SCHOLARSEEK_LOCAL_DATASET_DIR") or "missing",
                    "local_corpus_files": os.getenv("SCHOLARSEEK_LOCAL_CORPUS_FILES") or "default",
                    "local_max_results": os.getenv("SCHOLARSEEK_LOCAL_MAX_RESULTS") or "default",
                    "reranker_path": config.reranker_path or "missing",
                }
            )
            return
        self._serve_static(parsed.path)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path not in {"/api/search", "/api/answer"}:
            self.send_error(404, "Not found")
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8") if length else "{}"
            payload = json.loads(body)
            if parsed.path == "/api/search":
                config = get_api_config()
                result = search_papers(
                    query=str(payload.get("query", "")).strip(),
                    planner=payload.get("planner", "heuristic"),
                    answer=payload.get("answer", "none"),
                    sources=payload.get("sources") or config.sources or "local,openalex,arxiv",
                    max_queries=int(payload.get("max_queries", 3)),
                    per_query=int(payload.get("per_query", 5)),
                    limit=int(payload.get("limit", 10)),
                    strategy=str(payload.get("strategy", "standard")),
                )
            else:
                result = synthesize_answer_for_papers(
                    query=str(payload.get("query", "")).strip(),
                    papers=payload.get("papers") or [],
                )
            self._json(result)
        except Exception as exc:
            self._json({"error": str(exc)}, status=500)

    def log_message(self, fmt, *args):
        print("[%s] %s" % (self.log_date_time_string(), fmt % args))

    def _serve_static(self, request_path: str):
        if request_path in {"", "/"}:
            file_path = FRONTEND_DIR / "index.html"
        else:
            relative = Path(unquote(request_path).lstrip("/"))
            file_path = (FRONTEND_DIR / relative).resolve()
            if FRONTEND_DIR.resolve() not in file_path.parents and file_path != FRONTEND_DIR.resolve():
                self.send_error(403, "Forbidden")
                return

        if not file_path.exists() or not file_path.is_file():
            self.send_error(404, "Not found")
            return

        content_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
        data = file_path.read_bytes()
        self.send_response(200)
        self._cors_headers()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json(self, payload, status: int = 200):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self._cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the ScholarSeek web UI and API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5174)
    args = parser.parse_args(argv)

    get_api_config()
    _prefer_web_compatible_reranker()
    server = ThreadingHTTPServer((args.host, args.port), ScholarSeekHandler)
    print(f"ScholarSeek web server: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def _prefer_web_compatible_reranker() -> None:
    fallback = os.getenv("SCHOLARSEEK_FALLBACK_RERANKER_PATH")
    active = os.getenv("SCHOLARSEEK_RERANKER_PATH")
    if not fallback or not active:
        return
    if Path(active, "compact_reranker.json").exists() or str(active).endswith("compact_reranker.json"):
        return
    if not (Path(fallback).is_dir() or Path(fallback).exists()):
        return
    try:
        import torch

        version = tuple(int(part) for part in torch.__version__.split("+", 1)[0].split(".")[:2])
        if version >= (2, 1):
            return
    except Exception:
        pass
    os.environ["SCHOLARSEEK_RERANKER_PATH"] = fallback
    print(f"[warn] Web server using compact reranker fallback: {fallback}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
