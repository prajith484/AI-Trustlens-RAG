"""CLI: ingest documents into Qdrant + BM25 + Neo4j graph."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ingestion.service import get_ingestion_service


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest trusted documents for AI TrustLens RAG")
    ap.add_argument("--path", required=True, help="File (.txt/.pdf/.html) or directory")
    ap.add_argument("--domain", default=None,
                    help="Override domain for all docs (else per-file Domain: header or 'general')")
    ap.add_argument("--recursive", action="store_true",
                    help="Descend into subdirectories (needed for documents/ + documents/medical/)")
    ap.add_argument("--source-name", default=None)
    args = ap.parse_args()
    svc = get_ingestion_service()
    overrides = {"domain": args.domain} if args.domain else {}
    if args.source_name:
        overrides["source_name"] = args.source_name
    results = svc.ingest_path(args.path, recursive=args.recursive, **overrides)
    print(json.dumps(results, indent=2))
    print(f"Total documents tracked: {len(svc.list_documents())}")
    # Release the Qdrant local-file lock so the API/eval can open it next.
    try:
        from app.vector_store.qdrant_store import reset_qdrant_store
        reset_qdrant_store()
    except Exception:
        pass


if __name__ == "__main__":
    main()
