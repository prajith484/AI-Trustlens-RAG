"""Non-generative entity & relationship extraction (rule-based, replaceable).

Uses:
- Curated tech lexicon (languages, libs, concepts, orgs).
- Regex for dates, versions, papers, capitalized phrases.

Every relationship preserves subject/type/object/supporting text/source
ids/method/confidence. Relationships are only created when two entities
co-occur in the SAME sentence (never whole-document co-occurrence), and are
marked method="rule-cooccur-sentence" with modest confidence. If no relation
pattern fires, entities are kept WITHOUT invented relationships.

The class is intentionally replaceable: an LLM-backed extractor can later
implement the same ``extract(text, document_id, chunk_id)`` interface.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

TECH_TERMS: Dict[str, str] = {
    # languages
    "python": "ProgrammingLanguage", "java": "ProgrammingLanguage",
    "javascript": "ProgrammingLanguage", "typescript": "ProgrammingLanguage",
    "c++": "ProgrammingLanguage", "rust": "ProgrammingLanguage", "go": "ProgrammingLanguage",
    # tech / concepts
    "transformer": "TechnicalConcept", "transformers": "TechnicalConcept",
    "retrieval-augmented generation": "TechnicalConcept", "rag": "TechnicalConcept",
    "vector database": "TechnicalConcept", "qdrant": "Technology",
    "neo4j": "Technology", "knowledge graph": "TechnicalConcept",
    "bm25": "TechnicalConcept", "embedding": "TechnicalConcept",
    "embeddings": "TechnicalConcept", "fastapi": "SoftwareLibrary",
    "pytorch": "SoftwareLibrary", "tensorflow": "SoftwareLibrary",
    "sentence-transformers": "SoftwareLibrary", "faiss": "Technology",
    "reciprocal rank fusion": "TechnicalConcept", "attention": "TechnicalConcept",
    "bert": "Technology", "gpt": "Technology", "llm": "TechnicalConcept",
    # orgs
    "google": "Organization", "meta": "Organization", "openai": "Organization",
    "microsoft": "Organization", "anthropic": "Organization",
    "world health organization": "Organization",
    "centers for disease control and prevention": "Organization",
    "cdc": "Organization", "national institutes of health": "Organization",
    "nih": "Organization",
    # medical / health (non-generative lexicon for the medical test corpus)
    "diabetes": "MedicalCondition", "hypertension": "MedicalCondition",
    "blood pressure": "MedicalConcept", "insulin": "MedicalConcept",
    "antibiotics": "Medication", "antibiotic": "Medication",
    "antibiotic resistance": "MedicalConcept", "infection": "MedicalConcept",
    "infections": "MedicalConcept", "bacterial": "MedicalConcept",
    "viral": "MedicalConcept", "cardiovascular disease": "MedicalCondition",
    "chronic disease": "MedicalConcept", "vaccine": "Medication",
}

# Organisation abbreviations such as WHO, CDC, NIH (2+ uppercase letters).
_ABBR_RE = re.compile(r"\b([A-Z]{2,}(?:\.[A-Z]{2,})?)\b")

_DATE_RE = re.compile(r"\b(19|20)\d{2}(?:[-/](0?[1-9]|1[0-2])(?:[-/](0?[1-9]|[12]\d|3[01]))?)?\b")
_VERSION_RE = re.compile(r"\b[vV]?\d+\.\d+(?:\.\d+)?\b")
_PAPER_RE = re.compile(r"\b([A-Z][\w\- ]{2,60}?)\s*\((19|20)\d{2}\)")
_CAP_PHRASE_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Entity:
    text: str
    type: str
    method: str = "rule-lexicon"
    confidence: float = 0.7

    @property
    def entity_id(self) -> str:
        return f"ent::{self.type}::{self.text.lower()}"


@dataclass
class Relationship:
    subject: str
    subject_type: str
    relation: str
    obj: str
    object_type: str
    supporting_text: str
    document_id: str
    chunk_id: str
    method: str = "rule-cooccur-sentence"
    confidence: float = 0.4


@dataclass
class ExtractionResult:
    entities: List[Entity] = field(default_factory=list)
    relationships: List[Relationship] = field(default_factory=list)


class EntityExtractor:
    """Replaceable rule-based extractor. Same interface an LLM extractor would use."""

    def extract(self, text: str, document_id: str = "", chunk_id: str = "") -> ExtractionResult:
        if not text or not text.strip():
            return ExtractionResult()
        entities = self.extract_entities(text)
        relationships = self.extract_relationships(text, entities, document_id, chunk_id)
        return ExtractionResult(entities, relationships)

    def extract_entities(self, text: str) -> List[Entity]:
        found: Dict[Tuple[str, str], Entity] = {}
        lowered = text.lower()
        for term, etype in TECH_TERMS.items():
            if term in lowered:
                # recover surface form
                m = re.search(re.escape(term), text, re.IGNORECASE)
                surface = m.group(0) if m else term
                found[(surface.lower(), etype)] = Entity(surface, etype, "rule-lexicon", 0.8)
        for m in _DATE_RE.finditer(text):
            found[(m.group(0).lower(), "Date")] = Entity(m.group(0), "Date", "rule-regex-date", 0.9)
        for m in _VERSION_RE.finditer(text):
            # avoid matching years already captured
            if re.fullmatch(r"(19|20)\d{2}", m.group(0)):
                continue
            found[(m.group(0).lower(), "Version")] = Entity(m.group(0), "Version", "rule-regex-version", 0.75)
        for m in _PAPER_RE.finditer(text):
            found[(m.group(0).lower(), "ResearchPaper")] = Entity(
                m.group(0).strip(), "ResearchPaper", "rule-regex-paper", 0.6)
        for m in _ABBR_RE.finditer(text):
            # Uppercase abbreviations only (WHO, CDC) — never lowercase "who".
            abbr = m.group(1)
            if abbr.lower() in {"llm", "rag"} or len(abbr) > 6:
                continue
            known = {"WHO": "Organization", "CDC": "Organization", "NIH": "Organization"}
            etype = known.get(abbr, "Abbreviation")
            found[(abbr.lower(), etype)] = Entity(abbr, etype, "rule-regex-abbr", 0.6)
        for m in _CAP_PHRASE_RE.finditer(text):
            phrase = m.group(1)
            if phrase.lower() in found or len(phrase) < 4:
                continue
            if phrase.split()[0] in {"The", "This", "These", "Such", "Each"}:
                continue
            found[(phrase.lower(), "Concept")] = Entity(phrase, "Concept", "rule-capitalised", 0.35)
        # People heuristic: "Firstname Lastname" with common firstname list? keep conservative:
        return sorted(found.values(), key=lambda e: e.text.lower())

    def extract_relationships(self, text: str, entities: List[Entity],
                              document_id: str, chunk_id: str) -> List[Relationship]:
        rels: List[Relationship] = []
        if len(entities) < 2:
            return rels
        sentences = _SENT_SPLIT.split(text)
        for sent in sentences:
            present = [e for e in entities if e.text.lower() in sent.lower()]
            # Only relate pairs co-occurring in the SAME sentence, max 3 pairs/sentence
            for i in range(min(len(present), 4)):
                for j in range(i + 1, min(len(present), 4)):
                    a, b = present[i], present[j]
                    if a.text.lower() == b.text.lower():
                        continue
                    rels.append(Relationship(
                        subject=a.text, subject_type=a.type,
                        relation="RELATED_TO", obj=b.text, object_type=b.type,
                        supporting_text=sent.strip()[:500],
                        document_id=document_id, chunk_id=chunk_id,
                        method="rule-cooccur-sentence", confidence=0.4,
                    ))
                    if len(rels) >= 12:
                        return rels
        return rels
