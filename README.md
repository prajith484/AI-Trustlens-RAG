# AI TrustLens — RAG & Graph RAG Evidence Retrieval Subsystem

Standalone **evidence retrieval service** for the AI TrustLens audit platform.
Given factual claims (from another teammate's claim-extraction module), it returns
**traceable source passages** — never verification labels, trust scores, or generated answers.

> Boundary: **no generative LLM** (no Ollama/OpenAI/Gemini/Claude), **no claim classifier**.
> The only model allowed is the `sentence-transformers/all-MiniLM-L6-v2` **embedding** model
> (vector representations, cosine similarity).

## Architecture

```
Claim ──┬──► Vector (embed claim → Qdrant cosine search) ──┐
        ├──► BM25 keyword search (local rank-bm25 index) ──├──► RRF fuse ──► rank/dedup ──► Evidence[]
        └──► Graph (entities → Neo4j / local snapshot traversal → chunks) ──┘
```

- `app/ingestion/` — TXT/PDF/HTML loaders (incl. `Source:/Title:/URL:` header parsing) +
  cleaning + paragraph-aware chunking (200w/30 overlap)
- `app/embeddings/` — SentenceTransformer singleton (hash fallback offline), cosine metric
- `app/vector_store/qdrant_store.py` — Qdrant local-file (default, persistent), `:memory:`, or server
- `app/retrieval/bm25_retriever.py` — local BM25, pickled to `BM25_INDEX_PATH`
- `app/graph_store/` — rule-based entity/relation extraction (tech + medical lexicon) +
  Neo4j store with JSON-snapshot fallback (`GRAPH_STORE_PATH`)
- `app/retrieval/hybrid_retriever.py` — RRF (k=60), modes `vector|bm25|graph|hybrid`,
  domain filtering enforced per-method, at ranking, and as a final check
- `app/api/` — Pydantic schemas (integration contract) + routes; `app/main.py` — FastAPI app
- `evaluation/` — Recall@k, Precision@k, MRR, latency, coverage + general/medical datasets
- `documents/` — sample corpus (`documents/medical/` holds the Medical test docs)

## Windows setup, no Docker (PowerShell, verified)

```powershell
# 1. Activate the venv
.\.venv\Scripts\Activate.ps1

# 2. Install deps
pip install -r requirements.txt

# 3. Configure local development (ALREADY DONE: .env points at local files)
Copy-Item .env.example .env   # QDRANT_URL=./qdrant_storage etc.

# 4. Ingest general + medical documents (STOP the API first — see note below)
python scripts/ingest_documents.py --path documents --domain general
python scripts/ingest_documents.py --path documents/medical --domain Medical
# --recursive also available for nested folders

# 5. Run the API
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

# 6. Test retrieval (new PowerShell window)
Invoke-RestMethod -Uri http://127.0.0.1:8000/health -Method Get
Invoke-RestMethod -Uri http://127.0.0.1:8000/retrieve -Method Post `
  -ContentType 'application/json' `
  -Body '{"claim": "Diabetes is a chronic disease.", "top_k": 3, "retrieval_mode": "hybrid", "domain": "Medical"}'

# 7. Restart check: stop API (Ctrl+C), start again, re-run health —
#    qdrant points / graph chunks / bm25 size must be unchanged.

# 8. Run tests / evaluations (STOP the API first — see note below)
python -m pytest
python evaluation/evaluate.py --docs documents --dataset evaluation/test_dataset.json --out evaluation/results.json
python evaluation/evaluate.py --docs documents/medical --dataset evaluation/medical_dataset.json `
  --ingest-domain Medical --out evaluation/medical_results.json
```

> **Sequential-access note:** Qdrant local-file mode takes an exclusive folder lock:
> only ONE process (ingest script, API, or eval) may use `./qdrant_storage` at a
> time. A second process gets a clear `... already accessed by another instance ...`
> error in diagnostics (reported, never silent). For concurrent access, run
> Qdrant server (`docker compose up -d`, `QDRANT_URL=http://localhost:6333`).

Adding documents: drop `.txt`/`.pdf`/`.html` into a folder and ingest it. A text file
may start with a provenance header (`Source:`, `Title:`, `URL:`, `Date:`, `Domain:`)
— headers become structured metadata (returned as `source_url` etc.) and are NOT
indexed as evidence text. CLI `--domain` overrides the header. Re-ingesting
unchanged content is a no-op (stable IDs via content hash).

Reset dev data (derived caches only — source `.txt`/PDFs are never touched):

```powershell
Remove-Item -Recurse -Force qdrant_storage, bm25_index.pkl, graph_store.json, documents_registry.json -ErrorAction SilentlyContinue
```

## Sample API

Medical-only hybrid request:

```json
{"claim": "Diabetes is a chronic disease.", "top_k": 3, "retrieval_mode": "hybrid", "domain": "Medical"}
```

Response (actual shape):

```json
{
  "claim": "Diabetes is a chronic disease.",
  "retrieval_mode": "hybrid",
  "evidence": [
    {"chunk_id": "doc_diabetes_e9d0a8e1::c0000::...", "document_id": "doc_diabetes_e9d0a8e1",
     "text": "Original passage...", "title": "Diabetes",
     "source_name": "World Health Organization (WHO)",
     "source_url": "https://www.who.int/news-room/fact-sheets/detail/diabetes",
     "domain": "Medical", "retrieval_methods": ["bm25", "graph", "vector"],
     "retrieval_score": 0.049, "graph_path": ["ent::MedicalConcept::chronic disease", "..."]}
  ],
  "evidence_found": true, "latency_ms": 20,
  "diagnostics": {"per_method": [{"method": "vector", "status": "ok", "count": 3, ...}], ...}
}
```

- `domain` (or `filters: {"domain": ...}`) restricts vector, BM25, graph, and final ranking.
- Per-method `status` is `ok` | `empty` | `error`: an error is a backend failure,
  `empty` is a successful search with no hits. `retrieval_score` is a ranking
  signal, NOT a truth probability.
- Other endpoints: `GET /health`, `POST /retrieve/batch`, `POST /documents/ingest`,
  `GET /documents`, `GET /documents/{id}`, `DELETE /documents/{id}`.

## Integration guide (LLM & classifier teammates)

1. Call `POST /retrieve` (or `/retrieve/batch` for lists) with each extracted claim.
2. You receive `evidence[]` with **original text + provenance** (`document_id`, `source_name`,
   `source_url`, `page_number`, `retrieval_methods`, `graph_path`).
3. Feed `claim + evidence[]` into your LLM/classifier; **do your own verification** —
   a high `retrieval_score` is similarity only, NOT truth.
4. `evidence_found=false` + `evidence=[]` means no support found (distinct from weak support).
5. Python shortcut: `from app.retrieval.hybrid_retriever import retrieve_hybrid_evidence, retrieve_evidence_for_claims`.

## Limitations

- Qdrant local-file mode: exclusive per-process lock (see note above).
- First graph-backend detection costs ~2s once per process on machines without Neo4j
  (cached afterwards; happens at API startup, not per request).
- Embedding model downloads (~80 MB) on first online run; offline falls back to deterministic
  hash embeddings (lower quality, flagged in health `embedding_backend` + eval config).
- Entity extraction is rule-based (tech + medical lexicon, regex); an LLM extractor can
  later implement the same `extract()` interface. Graph connectivity is never proof of truth.
- Missing metadata stays `null` — never invented (e.g. general docs have no `source_url`).
- No claim extraction, no LLM answers, no verification labels — by design.
