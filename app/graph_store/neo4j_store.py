"""Neo4j graph store with transparent in-memory fallback.

Schema:
  (:Document {document_id,...})-[:CONTAINS]->(:Chunk {chunk_id, text, ...})
  (:Chunk)-[:MENTIONS]->(:Entity {entity_id, text, type})
  (:Entity)-[:RELATED_TO {supporting_text, chunk_id, ...}]->(:Entity)
  (:Entity)-[:APPEARS_IN]->(:Document)

- Stable IDs, MERGE semantics (no duplicates), parameterized Cypher only.
- If Neo4j is unreachable/misconfigured, an in-memory graph with the same
  API is used so development/tests work without Docker. health_check()
  reports which backend is active.
- No arbitrary Cypher execution is exposed via the API.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Set

from .entity_extractor import Entity, Relationship


class _MemoryGraph:
    def __init__(self):
        self.documents: Dict[str, Dict] = {}
        self.chunks: Dict[str, Dict] = {}
        self.entities: Dict[str, Dict] = {}
        self.contains: Dict[str, Set[str]] = defaultdict(set)  # doc -> chunks
        self.chunk_entities: Dict[str, Set[str]] = defaultdict(set)  # chunk -> entities
        self.entity_docs: Dict[str, Set[str]] = defaultdict(set)
        self.related: Dict[str, List[Dict]] = defaultdict(list)  # ent_id -> [edge]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "documents": self.documents,
            "chunks": self.chunks,
            "entities": self.entities,
            "contains": {k: sorted(v) for k, v in self.contains.items()},
            "chunk_entities": {k: sorted(v) for k, v in self.chunk_entities.items()},
            "entity_docs": {k: sorted(v) for k, v in self.entity_docs.items()},
            "related": dict(self.related),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "_MemoryGraph":
        g = cls()
        g.documents = dict(data.get("documents", {}))
        g.chunks = dict(data.get("chunks", {}))
        g.entities = dict(data.get("entities", {}))
        g.contains = defaultdict(set, {k: set(v) for k, v in data.get("contains", {}).items()})
        g.chunk_entities = defaultdict(
            set, {k: set(v) for k, v in data.get("chunk_entities", {}).items()})
        g.entity_docs = defaultdict(
            set, {k: set(v) for k, v in data.get("entity_docs", {}).items()})
        g.related = defaultdict(list, data.get("related", {}))
        return g

    def merge_document(self, meta: Dict) -> None:
        self.documents[meta["document_id"]] = dict(meta)

    def merge_chunk(self, payload: Dict) -> None:
        self.chunks[payload["chunk_id"]] = dict(payload)
        self.contains[payload["document_id"]].add(payload["chunk_id"])

    def merge_entity(self, ent: Entity, chunk_id: str, document_id: str) -> None:
        self.entities[ent.entity_id] = {"entity_id": ent.entity_id, "text": ent.text, "type": ent.type}
        self.chunk_entities[chunk_id].add(ent.entity_id)
        self.entity_docs[ent.entity_id].add(document_id)

    def merge_relationship(self, rel: Relationship) -> None:
        sid = f"ent::{rel.subject_type}::{rel.subject.lower()}"
        oid = f"ent::{rel.object_type}::{rel.obj.lower()}"
        for eid, txt, typ in ((sid, rel.subject, rel.subject_type), (oid, rel.obj, rel.object_type)):
            self.entities.setdefault(eid, {"entity_id": eid, "text": txt, "type": typ})
        edge = {"to": oid, "relation": rel.relation, "supporting_text": rel.supporting_text,
                "chunk_id": rel.chunk_id, "document_id": rel.document_id,
                "confidence": rel.confidence, "method": rel.method}
        if not any(e["to"] == oid and e["chunk_id"] == rel.chunk_id for e in self.related[sid]):
            self.related[sid].append(edge)
        self.entity_docs[sid].add(rel.document_id)
        self.entity_docs[oid].add(rel.document_id)

    def delete_document(self, document_id: str) -> None:
        chunks = self.contains.pop(document_id, set())
        for cid in chunks:
            self.chunks.pop(cid, None)
            self.chunk_entities.pop(cid, None)
        self.documents.pop(document_id, None)
        # prune edges referencing deleted chunks
        for sid in list(self.related):
            self.related[sid] = [e for e in self.related[sid] if e["document_id"] != document_id]
        for eid in list(self.entity_docs):
            self.entity_docs[eid].discard(document_id)
        # drop orphan entities (no remaining documents) and dangling edges
        orphaned = {eid for eid, docs in self.entity_docs.items() if not docs}
        for eid in orphaned:
            self.entity_docs.pop(eid, None)
            self.entities.pop(eid, None)
            self.related.pop(eid, None)
        for sid in list(self.related):
            self.related[sid] = [e for e in self.related[sid] if e["to"] not in orphaned]
        # remove pruned entities from chunk links
        for cid in list(self.chunk_entities):
            self.chunk_entities[cid] -= orphaned

    def find_entities(self, names: List[str]) -> List[Dict]:
        lowered = [n.lower() for n in names]
        out = []
        for eid, ent in self.entities.items():
            t = ent["text"].lower()
            if any(n in t or t in n for n in lowered):
                out.append(ent)
        return out

    def traverse(self, start_ids: List[str], max_depth: int = 2) -> List[Dict]:
        """BFS over RELATED_TO; returns edges with paths."""
        results: List[Dict] = []
        visited: Set[str] = set(start_ids)
        queue: deque = deque([(sid, 0, [sid]) for sid in start_ids])
        while queue:
            cur, depth, path = queue.popleft()
            if depth >= max_depth:
                continue
            for edge in self.related.get(cur, []):
                nxt = edge["to"]
                epath = path + [nxt]
                results.append({"from": cur, "to": nxt, "path": epath,
                                "depth": depth + 1, **edge})
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, depth + 1, epath))
        return results


class Neo4jStore:
    def __init__(self, uri: str = "bolt://localhost:7687", username: str = "neo4j",
                 password: str = "trustlens", snapshot_path: str | None = None,
                 timeout: int = 5):
        self.uri = uri
        self.username = username
        self.password = password
        self.timeout = timeout
        if snapshot_path is None:
            try:
                from app.config import settings
                snapshot_path = settings.graph_store_path
            except Exception:
                snapshot_path = "./graph_store.json"
        self.snapshot_path = snapshot_path or ""
        self._driver = None
        self._mem = _MemoryGraph()
        self._backend = "uninitialised"  # "neo4j" | "memory"
        self._connect_error: str | None = None
        self._load_snapshot()

    # -- snapshot persistence (memory fallback survives restarts) --------
    def _load_snapshot(self) -> int:
        if not self.snapshot_path:
            return 0
        try:
            from pathlib import Path
            import json
            p = Path(self.snapshot_path)
            if not p.exists():
                return 0
            self._mem = _MemoryGraph.from_dict(json.loads(p.read_text(encoding="utf-8")))
            return len(self._mem.chunks)
        except Exception as exc:
            print(f"[graph] snapshot load failed: {exc}")
            return 0

    def _save_snapshot(self) -> None:
        if not self.snapshot_path or self._backend == "neo4j":
            return  # Neo4j itself is the persistent store
        try:
            from pathlib import Path
            import json
            p = Path(self.snapshot_path)
            if p.parent != Path(".") and str(p.parent):
                p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(self._mem.to_dict()), encoding="utf-8")
        except Exception as exc:
            print(f"[graph] snapshot save failed: {exc}")

    # -- connection -----------------------------------------------------
    def _connect(self):
        if self._backend != "uninitialised":
            return self._driver if self._backend == "neo4j" else None
        try:
            from neo4j import GraphDatabase  # type: ignore
            if not self._port_open():
                raise ConnectionError(
                    f"Neo4j unreachable at {self.uri} "
                    f"(no listener; using local graph fallback)")
            driver = GraphDatabase.driver(
                self.uri, auth=(self.username, self.password),
                connection_timeout=self.timeout)
            driver.verify_connectivity()
            self._driver = driver
            self._backend = "neo4j"
            self._init_constraints()
            return driver
        except Exception as exc:
            self._driver = None
            self._backend = "memory"
            self._connect_error = str(exc)
            return None

    def _port_open(self) -> bool:
        """Fast pre-check: is anything listening? Avoids driver retry backoff.

        Single connection attempt only (cached per process afterwards): for
        `localhost` this targets 127.0.0.1 directly, because on some Windows
        setups each refused loopback connection (v6 or v4) can stall ~2s.
        """
        try:
            from urllib.parse import urlparse
            import socket
            parts = urlparse(self.uri)
            host = parts.hostname or "localhost"
            port = parts.port or 7687
            if host.lower() in ("localhost",):
                host = "127.0.0.1"
            with socket.create_connection((host, port),
                                          timeout=min(self.timeout, 2)):
                return True
        except Exception:
            return False

    @property
    def backend(self) -> str:
        self._connect()
        return self._backend

    def health_check(self) -> Dict[str, Any]:
        self._connect()
        counts = {"documents": len(self._mem.documents), "chunks": len(self._mem.chunks),
                  "entities": len(self._mem.entities)}
        if self._backend == "neo4j":
            return {"ok": True, "backend": "neo4j", "persistent": True,
                    "uri": self.uri, "counts": counts, "error": None}
        return {"ok": True, "backend": "memory-fallback", "persistent": bool(self.snapshot_path),
                "uri": self.uri, "snapshot_path": self.snapshot_path or None,
                "counts": counts, "error": None,
                "connect_error": self._connect_error,
                "note": "Neo4j unavailable; using local graph fallback (JSON snapshot)."}

    def _init_constraints(self) -> None:
        stmts = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.document_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE",
        ]
        assert self._driver is not None
        with self._driver.session() as s:
            for st in stmts:
                try:
                    s.run(st)
                except Exception:
                    pass

    # -- writes ----------------------------------------------------------
    def merge_document(self, meta: Dict) -> None:
        self._mem.merge_document(meta)
        self._save_snapshot()
        drv = self._connect()
        if drv is None:
            return
        with drv.session() as s:
            s.run(
                "MERGE (d:Document {document_id: $document_id}) "
                "SET d.title=$title, d.source_name=$source_name, d.source_url=$url, "
                "d.domain=$domain, d.publication_date=$pub, d.content_hash=$ch",
                document_id=meta.get("document_id"), title=meta.get("title"),
                source_name=meta.get("source_name"), url=meta.get("source_url"),
                domain=meta.get("domain"),
                pub=meta.get("publication_date"), ch=meta.get("content_hash"))

    def merge_chunk(self, payload: Dict) -> None:
        self._mem.merge_chunk(payload)
        self._save_snapshot()
        drv = self._connect()
        if drv is None:
            return
        with drv.session() as s:
            s.run(
                "MERGE (c:Chunk {chunk_id: $cid}) "
                "SET c.text=$text, c.document_id=$did, c.page_number=$page, "
                "c.chunk_position=$pos, c.title=$title, c.source_name=$sname, "
                "c.source_url=$url, c.domain=$domain, c.publication_date=$pub "
                "WITH c MATCH (d:Document {document_id:$did}) "
                "MERGE (d)-[:CONTAINS]->(c)",
                cid=payload.get("chunk_id"), text=payload.get("text"),
                did=payload.get("document_id"), page=payload.get("page_number"),
                pos=payload.get("chunk_position"), title=payload.get("title"),
                sname=payload.get("source_name"), url=payload.get("source_url"),
                domain=payload.get("domain"), pub=payload.get("publication_date"))

    def merge_entity(self, ent: Entity, chunk_id: str, document_id: str) -> None:
        self._mem.merge_entity(ent, chunk_id, document_id)
        self._save_snapshot()
        drv = self._connect()
        if drv is None:
            return
        with drv.session() as s:
            s.run(
                "MERGE (e:Entity {entity_id:$eid}) SET e.text=$text, e.type=$type "
                "WITH e MATCH (c:Chunk {chunk_id:$cid}) "
                "MERGE (c)-[:MENTIONS]->(e) "
                "WITH e MATCH (d:Document {document_id:$did}) "
                "MERGE (e)-[:APPEARS_IN]->(d)",
                eid=ent.entity_id, text=ent.text, type=ent.type,
                cid=chunk_id, did=document_id)

    def merge_relationship(self, rel: Relationship) -> None:
        self._mem.merge_relationship(rel)
        self._save_snapshot()
        drv = self._connect()
        if drv is None:
            return
        sid = f"ent::{rel.subject_type}::{rel.subject.lower()}"
        oid = f"ent::{rel.object_type}::{rel.obj.lower()}"
        with drv.session() as s:
            s.run(
                "MATCH (a:Entity {entity_id:$sid}), (b:Entity {entity_id:$oid}) "
                "MERGE (a)-[r:RELATED_TO {chunk_id:$cid}]->(b) "
                "SET r.supporting_text=$st, r.document_id=$did, "
                "r.confidence=$conf, r.method=$method",
                sid=sid, oid=oid, cid=rel.chunk_id, st=rel.supporting_text,
                did=rel.document_id, conf=rel.confidence, method=rel.method)

    def delete_document(self, document_id: str) -> None:
        self._mem.delete_document(document_id)
        self._save_snapshot()
        drv = self._connect()
        if drv is None:
            return
        with drv.session() as s:
            s.run(
                "MATCH (d:Document {document_id:$did}) OPTIONAL MATCH (d)-[:CONTAINS]->(c:Chunk) "
                "DETACH DELETE c, d", did=document_id)

    # -- reads -----------------------------------------------------------
    def find_entities(self, names: List[str]) -> List[Dict]:
        drv = self._connect()
        mem_hits = self._mem.find_entities(names)
        if drv is None or not names:
            return mem_hits
        out: List[Dict] = []
        with drv.session() as s:
            for n in names[:10]:
                recs = s.run(
                    "MATCH (e:Entity) WHERE toLower(e.text) CONTAINS toLower($n) "
                    "RETURN e.entity_id AS entity_id, e.text AS text, e.type AS type LIMIT 5",
                    n=n)
                out.extend([dict(r) for r in recs])
        # merge, dedup by entity_id
        seen = {e["entity_id"]: e for e in mem_hits}
        for e in out:
            seen.setdefault(e["entity_id"], e)
        return list(seen.values())

    def traverse(self, start_ids: List[str], max_depth: int = 2) -> List[Dict]:
        mem_edges = self._mem.traverse(start_ids, max_depth)
        drv = self._connect()
        if drv is None:
            return mem_edges
        out: List[Dict] = []
        with drv.session() as s:
            for sid in start_ids[:10]:
                recs = s.run(
                    "MATCH p=(a:Entity {entity_id:$sid})-[r:RELATED_TO*1..2]->(b:Entity) "
                    "RETURN [n IN nodes(p) | n.entity_id] AS path, "
                    "r[-1].chunk_id AS chunk_id, r[-1].document_id AS document_id, "
                    "r[-1].supporting_text AS supporting_text, length(p) AS depth LIMIT 25",
                    sid=sid)
                for r in recs:
                    out.append({"from": sid, "to": r["path"][-1], "path": list(r["path"]),
                                "depth": int(r["depth"]), "chunk_id": r["chunk_id"],
                                "document_id": r["document_id"],
                                "supporting_text": r["supporting_text"]})
        # merge mem + neo4j, dedup by (from,to,chunk_id)
        seen = {(e["from"], e["to"], e.get("chunk_id")): e for e in mem_edges}
        for e in out:
            seen.setdefault((e["from"], e["to"], e.get("chunk_id")), e)
        return list(seen.values())

    def chunks_for_entities(self, entity_ids: List[str],
                            filters: Optional[Dict[str, Any]] = None) -> List[Dict]:
        """Return chunk payloads linked to entities, honouring domain filters.

        Uses the Neo4j server when connected (so data written by an earlier
        process is visible); otherwise the local mirror/snapshot.
        """
        domain = (filters or {}).get("domain")

        def _keep(ch: Dict) -> bool:
            return domain is None or ch.get("domain") == domain

        drv = self._connect()
        if drv is not None:
            try:
                with drv.session() as s:
                    recs = s.run(
                        "MATCH (c:Chunk)-[:MENTIONS]->(e:Entity) "
                        "WHERE e.entity_id IN $ids "
                        "RETURN DISTINCT c.chunk_id AS chunk_id, c.text AS text, "
                        "c.document_id AS document_id, c.title AS title, "
                        "c.source_name AS source_name, c.source_url AS source_url, "
                        "c.domain AS domain, c.publication_date AS publication_date, "
                        "c.page_number AS page_number, c.chunk_position AS chunk_position",
                        ids=entity_ids[:50])
                    return [dict(r) for r in recs if _keep(dict(r))]
            except Exception:
                pass  # fall through to the local mirror
        out = []
        for cid, eids in self._mem.chunk_entities.items():
            if any(e in entity_ids for e in eids):
                ch = self._mem.chunks.get(cid)
                if ch and _keep(ch):
                    out.append(ch)
        return out


# Process-wide singleton: ingestion and retrieval must share one graph,
# otherwise the memory fallback would diverge between writer and reader.
_singleton: "Neo4jStore | None" = None


def get_graph_store(uri: str | None = None, username: str | None = None,
                    password: str | None = None,
                    snapshot_path: str | None = None) -> "Neo4jStore":
    global _singleton
    if _singleton is None:
        from app.config import settings
        _singleton = Neo4jStore(
            uri=uri or settings.neo4j_uri,
            username=username or settings.neo4j_username,
            password=password or settings.neo4j_password,
            snapshot_path="" if snapshot_path == "" else (snapshot_path or settings.graph_store_path),
            timeout=settings.neo4j_timeout)
    return _singleton


def reset_graph_store() -> None:
    global _singleton
    _singleton = None
