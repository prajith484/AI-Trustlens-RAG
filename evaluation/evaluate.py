"""Run retrieval evaluation across modes; save JSON + CSV.

Compares: vector | bm25 | graph | hybrid. Only reports metrics actually
executed. Records the active embedding backend + retrieval configuration, so
results are never mistaken for a different setup (e.g. hash-fallback numbers
must not be presented as MiniLM performance).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ingestion.service import IngestionService, reset_ingestion_service
from app.retrieval.bm25_retriever import reset_bm25_index
from evaluation.metrics import summarize


def load_dataset(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def evaluate_mode(service, claims_data: list[dict], mode: str, top_k: int = 5,
                  default_domain: str | None = None) -> dict:
    from app.retrieval.bm25_retriever import get_bm25_index
    from app.retrieval.hybrid_retriever import retrieve_hybrid_evidence
    per_claim = []
    for item in claims_data:
        claim = item["claim"]
        domain = item.get("domain", default_domain)
        filters = {"domain": domain} if domain else None
        relevant_texts = item.get("relevant_markers", [])
        # Resolve relevant chunk ids against indexed payloads (same domain).
        relevant_ids = set()
        idx = get_bm25_index()
        for marker in relevant_texts:
            for p in idx.payloads:
                if domain and p.get("domain") != domain:
                    continue
                if marker.lower() in p.get("text", "").lower():
                    relevant_ids.add(p["chunk_id"])
        res = retrieve_hybrid_evidence(claim, top_k=top_k, retrieval_mode=mode,
                                       filters=filters)
        retrieved_ids = [e["chunk_id"] for e in res["evidence"]]
        retrieved_domains = sorted({e.get("domain") for e in res["evidence"]})
        excluded = item.get("must_exclude_domains", [])
        leaked = sorted(set(retrieved_domains) & set(excluded))
        hits = len(set(retrieved_ids) & relevant_ids)
        per_claim.append({"id": item.get("id"), "kind": item.get("kind"),
                          "claim": claim, "domain_filter": domain,
                          "retrieved": retrieved_ids, "retrieved_domains": retrieved_domains,
                          "relevant": sorted(relevant_ids), "hits": hits,
                          "excluded_domain_leak": leaked,
                          "latency_ms": res.get("latency_ms", 0),
                          "evidence_found": res.get("evidence_found", False),
                          "diagnostics": res.get("diagnostics", {})})
    return {"mode": mode, "domain": default_domain,
            "summary": summarize(per_claim), "per_claim": per_claim}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="documents")
    ap.add_argument("--dataset", default="evaluation/test_dataset.json")
    ap.add_argument("--domain", default=None,
                    help="Default domain filter (also used for ingest override)")
    ap.add_argument("--ingest-domain", default=None,
                    help="Domain override applied at ingest time (e.g. Medical)")
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--out", default="evaluation/results.json")
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    reset_bm25_index()
    reset_ingestion_service()
    from app.config import settings
    service = IngestionService()
    print(f"[eval] ingesting {args.docs} (recursive={args.recursive}) ...")
    ingest_overrides = {"domain": args.ingest_domain} if args.ingest_domain else {}
    results = service.ingest_path(args.docs, recursive=args.recursive, **ingest_overrides)
    print(f"[eval] ingest results: {results}")
    data = load_dataset(args.dataset)
    claims = data["claims"]
    default_domain = args.domain or data.get("default_domain")

    from app.embeddings.embedding_service import get_embedding_service
    emb = get_embedding_service()
    try:
        emb_dim = emb.dimension()
    except Exception:
        emb_dim = settings.embedding_dim
    run_config = {
        "embedding_backend": emb.backend, "embedding_model": emb.model_name,
        "embedding_dim": emb_dim, "qdrant_url": settings.qdrant_url,
        "qdrant_collection": settings.qdrant_collection,
        "retrieval_modes": ["vector", "bm25", "graph", "hybrid"],
        "note": ("Metrics measure RETRIEVAL quality (recall/MRR/coverage), "
                 "not medical fact-verification accuracy."),
    }
    modes = ["vector", "bm25", "graph", "hybrid"]
    output = {"top_k": args.top_k, "config": run_config, "modes": {}}
    for mode in modes:
        print(f"[eval] mode={mode} ...")
        try:
            output["modes"][mode] = evaluate_mode(service, claims, mode,
                                                  top_k=args.top_k,
                                                  default_domain=default_domain)
        except Exception as exc:
            output["modes"][mode] = {"mode": mode, "error": str(exc)}
    Path(args.out).write_text(json.dumps(output, indent=2), encoding="utf-8")
    csv_path = Path(args.out).with_suffix(".csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["mode", "recall@1", "recall@3", "recall@5",
                    "precision@5", "mrr", "mean_latency_ms", "coverage",
                    "domain_leaks"])
        for mode, m in output["modes"].items():
            s = m.get("summary", {})
            leaks = sum(1 for c in m.get("per_claim", []) if c.get("excluded_domain_leak"))
            w.writerow([mode, s.get("recall@1"), s.get("recall@3"), s.get("recall@5"),
                        s.get("precision@5"), s.get("mrr"),
                        s.get("mean_latency_ms"), s.get("coverage"), leaks])
    print(f"[eval] wrote {args.out} and {csv_path}")
    print(json.dumps({m: v.get("summary", v.get("error")) for m, v in output["modes"].items()}, indent=2))
    # Release the Qdrant local-file lock for the next process.
    try:
        from app.vector_store.qdrant_store import reset_qdrant_store
        reset_qdrant_store()
    except Exception:
        pass


if __name__ == "__main__":
    main()
