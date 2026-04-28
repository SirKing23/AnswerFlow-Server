"""
entity_extractor.py
====================
Extracts named entities and their relationships from text chunks
using GPT-4o-mini with structured JSON output.

Drop this file into:  services/entity_extractor.py

Education-domain aware — classifies each document into one of 5 profiles
and uses a specialized extraction prompt per profile for higher accuracy.

Profiles:
  policy      — Memos, issuances, DepEd orders, laws, bylaws
  curriculum  — Curriculum guides, MELCs, lesson plans, pacing guides
  performance — Grading sheets, test results, student work samples
  report      — Accomplishment reports, narrative reports
  content     — Stories, songs, e-learning materials, handouts, presentations

Called by pipelineDocument.py once per chunk, right after embedding.

Signature change from the original:
  extract_entities_and_relations(chunk_text, file_name, metadata) → dict

Returns:
  {
    "entities":  [{"name": "DepEd Order 42 s.2024", "type": "POLICY"}, ...],
    "relations": [{"source": "DepEd Order 42 s.2024", "target": "K-12 Program",
                   "type": "APPLIES_TO"}, ...]
  }
"""

import sys
import os
import json
import logging

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from openai import AsyncOpenAI
from config import (
    OPENAI_API_KEY,
    ENTITY_EXTRACTOR_TEMPERATURE, ENTITY_EXTRACTOR_MAX_TOKENS,
    ENTITY_CAP_DEFAULT, ENTITY_CAP_CONTENT,
    RELATION_CAP_DEFAULT, RELATION_CAP_CONTENT,
    ENTITY_EXTRACT_CHAR_LIMIT,
    # Per-profile extraction prompts — defined in .env, assembled in config.py
    # Each prompt includes ENTITY_BASE_RULES (injected via <<BASE_RULES>> at startup)
    ENTITY_PROMPT_POLICY,
    ENTITY_PROMPT_CURRICULUM,
    ENTITY_PROMPT_PERFORMANCE,
    ENTITY_PROMPT_REPORT,
    ENTITY_PROMPT_CONTENT,
)

log = logging.getLogger(__name__)

_client = AsyncOpenAI(api_key=OPENAI_API_KEY)


# ── Entity types ───────────────────────────────────────────────────────────────

ENTITY_TYPES = [
    # People
    "STUDENT",      # Named learners in grading sheets, reports, student work
    "TEACHER",      # Named teaching staff
    "SCHOOL_HEAD",  # Principals, head teachers, master teachers
    "OFFICIAL",     # Superintendents, directors, DepEd officials, signatories

    # Institutions
    "SCHOOL",       # Named schools e.g. "Olongapo City National High School"
    "DISTRICT",     # School districts e.g. "Subic District"
    "DIVISION",     # School divisions e.g. "SDO Olongapo City"
    "REGION",       # DepEd regional offices e.g. "DepEd Region III"

    # Curriculum
    "SUBJECT",      # Learning areas: Math, Filipino, Science, EPP, MAPEH
    "GRADE_LEVEL",  # Kinder, Grade 1-12, SHS strand e.g. "Grade 4", "ABM"
    "COMPETENCY",   # Specific MELC codes and learning objectives
    "PROGRAM",      # K-12, SHS, ALS, Brigada Eskwela, Marungko, MTB-MLE

    # Policy & Legal
    "POLICY",       # DepEd Orders, Memos, Circulars, Republic Acts, bylaws
    "PROVISION",    # Specific numbered sections within a law or memo

    # Assessment & Performance
    "ASSESSMENT",   # Quarterly exams, PTs, NAT, PISA, formative tasks
    "METRIC",       # Scores, MPS, percentages, enrollment figures, ratings

    # Time
    "PERIOD",       # School year, quarter, grading period e.g. "SY 2023-2024", "Q3"

    # Content materials only
    "CONCEPT",      # Abstract ideas, topics, themes in teaching materials
    "CHARACTER",    # Named characters in stories or reading materials
    "SETTING",      # Named places or settings in stories or discussions
]


# ── Document profile classifier ────────────────────────────────────────────────

# Filename keyword signals per profile — checked first (fast, no LLM needed)
_FILENAME_SIGNALS: dict[str, list[str]] = {
    "policy": [
        "memo", "memorandum", "order", "circular", "directive", "issuance",
        "policy", "republic act", "ra ", "deped order", "do ", "division memo",
        "regional memo", "national memo", "district memo", "bylaw", "by-law",
        "magna carta", "child protection", "batas", "resolusyon", "resolution",
    ],
    "curriculum": [
        "melc", "curriculum", "competency", "competencies", "syllabus",
        "lesson plan", "dll", "dlp", "daily log", "pacing", "guide",
        "learning objective", "learning area", "cg ", "curriculum guide",
    ],
    "performance": [
        "grading", "grade sheet", "class record", "sf9", "sf10", "report card",
        "test result", "exam result", "score", "nat result", "quarterly exam",
        "assessment result", "performance", "student record",
    ],
    "report": [
        "accomplishment", "narrative report", "program report", "activity report",
        "annual report", "monthly report", "school report", "ipcrf", "rpms",
        "self assessment", "sar", "summary report",
    ],
    "content": [
        "story", "kwento", "song", "awit", "poem", "tula", "module", "activity sheet",
        "worksheet", "handout", "presentation", "powerpoint", "visual aid",
        "reading material", "learning material", "e-learning", "elearning",
        "discussion", "lecture", "flashcard", "game", "puzzle",
    ],
}

# element_types composition signals — used when filename is ambiguous
_ELEMENT_TYPE_SIGNALS: dict[str, list[str]] = {
    "performance": ["table"],
    "policy":      ["heading"],
    "content":     ["narrative_text", "list_item"],
}


def classify_document(file_name: str, metadata: dict) -> str:
    """
    Classify a document into one of 5 profiles without an LLM call.

    Priority:
      1. Filename keyword match  (most reliable)
      2. element_types composition from metadata
      3. Default to 'content'   (safest fallback for teaching materials)

    Args:
        file_name: Original filename e.g. "DepEd Memo 042 s.2024.pdf"
        metadata:  Chunk metadata dict with optional keys:
                     element_types, headings, section

    Returns:
        One of: "policy", "curriculum", "performance", "report", "content"
    """
    name_lower = file_name.lower()

    # 1. Filename signals — check each profile in priority order
    priority_order = ["policy", "performance", "curriculum", "report", "content"]
    for profile in priority_order:
        if any(signal in name_lower for signal in _FILENAME_SIGNALS[profile]):
            log.debug(f"[entity_extractor] '{file_name}' → profile={profile} (filename match)")
            return profile

    # 2. element_types composition
    element_types = metadata.get("element_types", [])
    if element_types:
        type_set = set(element_types)
        if "table" in type_set:
            log.debug(f"[entity_extractor] '{file_name}' → profile=performance (table elements)")
            return "performance"
        # Heading-heavy without table → likely policy or curriculum
        heading_count = element_types.count("heading")
        text_count    = element_types.count("text") + element_types.count("narrative_text")
        if heading_count > 2 and text_count > heading_count:
            log.debug(f"[entity_extractor] '{file_name}' → profile=policy (heading-heavy)")
            return "policy"

    # 3. Default
    log.debug(f"[entity_extractor] '{file_name}' → profile=content (default fallback)")
    return "content"


# ── Extraction prompts per profile ─────────────────────────────────────────────

# Prompts loaded from .env via config.py.
# ENTITY_BASE_RULES (shared JSON rules) is injected into each profile via
# the <<BASE_RULES>> placeholder, which config.py replaces at startup.
# Edit ENTITY_PROMPT_POLICY / _CURRICULUM / _PERFORMANCE / _REPORT / _CONTENT
# and ENTITY_BASE_RULES in .env to add entity/relation types without code changes.
_PROMPTS: dict[str, str] = {
    "policy":      ENTITY_PROMPT_POLICY,       # Used below in extract_entities_and_relations
    "curriculum":  ENTITY_PROMPT_CURRICULUM,   # Used below in extract_entities_and_relations
    "performance": ENTITY_PROMPT_PERFORMANCE,  # Used below in extract_entities_and_relations
    "report":      ENTITY_PROMPT_REPORT,        # Used below in extract_entities_and_relations
    "content":     ENTITY_PROMPT_CONTENT,       # Used below in extract_entities_and_relations
}


# ── Main extraction function ───────────────────────────────────────────────────

async def extract_entities_and_relations(
    chunk_text: str,
    file_name:  str = "",
    metadata:   dict | None = None,
) -> dict:
    """
    Run NER + relation extraction on a single text chunk.

    Classifies the document type from file_name and metadata, then uses
    a specialized prompt for that document profile. Falls back to empty
    result on any error so the main pipeline never fails because of a
    graph extraction issue.

    Args:
        chunk_text: The raw text of one chunk (ideally ≤ 2000 chars)
        file_name:  Original filename — used for document profile classification
        metadata:   Chunk metadata dict — used for profile classification fallback
                    Expected keys: element_types, headings, section

    Returns:
        {"entities": [...], "relations": [...]}
    """
    if not chunk_text or not chunk_text.strip():
        return {"entities": [], "relations": []}

    if metadata is None:
        metadata = {}

    # Truncate to avoid token waste on very large chunks
    text_input = chunk_text[:ENTITY_EXTRACT_CHAR_LIMIT].strip()

    # Classify document type and select prompt
    # Prompt: _PROMPTS[profile] — one of ENTITY_PROMPT_POLICY / _CURRICULUM / _PERFORMANCE /
    # _REPORT / _CONTENT, each loaded from .env via config.py (ENTITY_BASE_RULES injected)
    profile = classify_document(file_name, metadata)
    system_prompt = (
        _PROMPTS[profile].rstrip()
        + "\n\nReturn only a valid JSON object with top-level keys 'entities' and 'relations'."
    )

    # Enrich the user message with available structural context
    headings = metadata.get("headings", [])
    section  = metadata.get("section", "")

    context_hint = ""
    if headings:
        context_hint += f"Document section headings: {' > '.join(headings)}\n"
    if section and section not in headings:
        context_hint += f"Section: {section}\n"
    if context_hint:
        context_hint = context_hint.strip() + "\n\n"

    user_message = (
        f"{context_hint}Extract the entities and relations from the following text and respond in JSON.\n\n"
        f"Text:\n{text_input}"
    )

    try:
        response = await _client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
            temperature=ENTITY_EXTRACTOR_TEMPERATURE,
            max_tokens=ENTITY_EXTRACTOR_MAX_TOKENS,
            response_format={"type": "json_object"},
        )

        raw    = response.choices[0].message.content.strip()
        parsed = json.loads(raw)

        # Determine entity cap based on profile
        entity_cap   = ENTITY_CAP_CONTENT if profile == "content" else ENTITY_CAP_DEFAULT
        relation_cap = RELATION_CAP_CONTENT if profile == "content" else RELATION_CAP_DEFAULT

        entities  = _validate_entities(parsed.get("entities", []),  cap=entity_cap)
        relations = _validate_relations(parsed.get("relations", []), entities, cap=relation_cap)

        log.debug(
            f"[entity_extractor] {file_name} | profile={profile} | "
            f"{len(entities)} entities, {len(relations)} relations "
            f"from chunk ({len(text_input)} chars)"
        )
        return {"entities": entities, "relations": relations}

    except json.JSONDecodeError as e:
        log.warning(f"[entity_extractor] JSON parse failed ({file_name}): {e}")
        return {"entities": [], "relations": []}

    except Exception as e:
        log.warning(f"[entity_extractor] Extraction failed ({file_name}): {e}")
        return {"entities": [], "relations": []}


# ── Validation helpers ─────────────────────────────────────────────────────────

def _validate_entities(raw: list, cap: int = 10) -> list[dict]:
    """
    Sanitize entity list — remove malformed entries and normalize types.
    """
    valid = []
    seen  = set()

    for item in raw:
        if not isinstance(item, dict):
            continue

        name  = str(item.get("name", "")).strip()
        etype = str(item.get("type", "CONCEPT")).strip().upper()

        # Skip empty, single-char, or pure-number names
        if not name or len(name) < 2 or name.isdigit():
            continue

        # Normalize type — reject anything not in our accepted list
        if etype not in ENTITY_TYPES:
            etype = "CONCEPT"

        # Deduplicate within this chunk
        key = (name.lower(), etype)
        if key in seen:
            continue
        seen.add(key)

        valid.append({"name": name, "type": etype})

    return valid[:cap]


def _validate_relations(raw: list, entities: list[dict], cap: int = 10) -> list[dict]:
    """
    Sanitize relation list — both endpoints must exist in extracted entities.
    """
    entity_names = {e["name"].lower() for e in entities}
    valid = []
    seen  = set()

    for item in raw:
        if not isinstance(item, dict):
            continue

        source   = str(item.get("source", "")).strip()
        target   = str(item.get("target", "")).strip()
        rel_type = (
            str(item.get("type", "RELATED_TO"))
            .strip()
            .upper()
            .replace(" ", "_")
        )

        if not source or not target or source == target:
            continue

        # Both endpoints must be in the extracted entity list
        if source.lower() not in entity_names or target.lower() not in entity_names:
            continue

        key = (source.lower(), target.lower(), rel_type)
        if key in seen:
            continue
        seen.add(key)

        valid.append({"source": source, "target": target, "type": rel_type})

    return valid[:cap]


# ── Batch helper ───────────────────────────────────────────────────────────────

async def extract_batch(
    chunks:    list[str],
    file_name: str = "",
    metadata:  dict | None = None,
) -> list[dict]:
    """
    Extract entities from multiple chunks concurrently.
    Limits concurrency to 5 to avoid rate-limit spikes.
    All chunks share the same file_name and metadata for profile classification.
    """
    import asyncio
    semaphore = asyncio.Semaphore(5)

    async def _extract_one(text: str) -> dict:
        async with semaphore:
            return await extract_entities_and_relations(text, file_name, metadata)

    return await asyncio.gather(*[_extract_one(c) for c in chunks])