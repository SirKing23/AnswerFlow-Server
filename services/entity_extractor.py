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

_BASE_RULES = """
Rules you MUST follow:
- Extract ONLY entities explicitly named or clearly identified in the text
- Entity names must be in their canonical/full form
- Skip pronouns, generic nouns ("the teacher", "the school"), and vague references
- Relationships must have both endpoints present in the extracted entities list
- Relationship type must be a single SCREAMING_SNAKE_CASE verb phrase
- Return ONLY valid JSON — no markdown fences, no explanation

JSON format:
{
  "entities":  [{"name": "string", "type": "ENTITY_TYPE"}, ...],
  "relations": [{"source": "entity name", "target": "entity name", "type": "RELATION_TYPE"}, ...]
}
"""

_PROMPTS: dict[str, str] = {

    "policy": f"""You are a named entity recognition system specialized in Philippine education policy documents.
Extract entities and relationships from DepEd memos, orders, circulars, Republic Acts, and school bylaws.

Valid entity types:
  POLICY     — DepEd Orders, Memos, Circulars, Republic Acts, bylaws (use full official name)
  PROVISION  — Specific numbered sections e.g. "Section 4.2", "Item 3.a"
  OFFICIAL   — Named signatories, directors, superintendents, secretaries
  DIVISION   — School divisions e.g. "SDO Olongapo City"
  REGION     — Regional offices e.g. "DepEd Region III - Central Luzon"
  DISTRICT   — School districts e.g. "Subic District"
  SCHOOL     — Named schools e.g. "Naugsol Integrated School", "Matain Elementary School", "Subic National High School", "Sto. Tomas High School"
  PROGRAM    — Named programs e.g. "K-12 Basic Education Program", "ALS"
  GRADE_LEVEL — Grade levels or school types this policy applies to
  PERIOD     — Effective dates, school years, deadlines

Valid relation types:
  ISSUED_BY     — POLICY issued by OFFICIAL or DIVISION or REGION
  SUPERSEDES    — POLICY replaces an older POLICY
  AMENDS        — POLICY modifies another POLICY
  IMPLEMENTS    — POLICY implements a higher POLICY or Republic Act
  APPLIES_TO    — POLICY applies to SCHOOL, PROGRAM, GRADE_LEVEL, or DIVISION
  REQUIRES      — POLICY requires a specific action or compliance item
  REFERENCES    — POLICY references another POLICY or legal instrument
  SIGNED_BY     — POLICY signed by OFFICIAL
  EFFECTIVE_ON  — POLICY effective on a PERIOD
  COVERS        — PROVISION covers a specific topic or PROGRAM

Limit to the 10 most important entities and 10 most important relations.
If nothing is extractable, return empty arrays.

{_BASE_RULES}""",

    "curriculum": f"""You are a named entity recognition system specialized in Philippine curriculum documents.
Extract entities and relationships from curriculum guides, MELCs, lesson plans, and pacing guides.

Valid entity types:
  SUBJECT     — Learning areas e.g. "Filipino", "Mathematics", "Science", "MAPEH", "EPP"
  GRADE_LEVEL — Grade levels e.g. "Grade 3", "Kinder", "Grade 11", "SHS"
  COMPETENCY  — Specific MELC codes or learning objectives (use exact text if short)
  PROGRAM     — Teaching approaches e.g. "Marungko Approach", "MTB-MLE", "SHS", "ALS"
  TEACHER     — Named teachers if mentioned
  SCHOOL      — Named schools if mentioned
  PERIOD      — School year, quarter, grading period
  ASSESSMENT  — Named assessment types e.g. "Quarterly Assessment", "Performance Task"

Valid relation types:
  COVERS          — SUBJECT covers a COMPETENCY
  PRESCRIBED_FOR  — COMPETENCY prescribed for a GRADE_LEVEL
  ALIGNED_WITH    — ASSESSMENT aligned with a COMPETENCY
  USES_APPROACH   — SUBJECT or COMPETENCY uses a PROGRAM or teaching approach
  SEQUENCED_AFTER — COMPETENCY follows another COMPETENCY
  TAUGHT_IN       — COMPETENCY taught in a specific PERIOD
  REQUIRES        — COMPETENCY requires mastery of another COMPETENCY

Limit to the 10 most important entities and 10 most important relations.
If nothing is extractable, return empty arrays.

{_BASE_RULES}""",

    "performance": f"""You are a named entity recognition system specialized in Philippine school performance documents.
Extract entities and relationships from grading sheets, class records, test results, and assessment reports.

Valid entity types:
  STUDENT     — Named learners (use full name as written)
  TEACHER     — Named class adviser or subject teacher
  SCHOOL      — Named school
  SUBJECT     — Learning area being assessed
  GRADE_LEVEL — Grade and section e.g. "Grade 4 - Mabini"
  ASSESSMENT  — Named exam or task e.g. "Q1 Quarterly Exam", "Performance Task 2"
  METRIC      — Scores, MPS, percentage, rating e.g. "MPS 78.5", "Average 82"
  PERIOD      — Grading period, quarter, school year e.g. "Q3 SY 2023-2024"

Valid relation types:
  SCORED_ON    — STUDENT scored a METRIC on an ASSESSMENT
  ENROLLED_IN  — STUDENT enrolled in a GRADE_LEVEL
  TEACHES      — TEACHER teaches a SUBJECT to a GRADE_LEVEL
  ASSESSED_IN  — ASSESSMENT given in a PERIOD
  BELONGS_TO   — GRADE_LEVEL belongs to a SCHOOL
  ACHIEVED     — GRADE_LEVEL or SCHOOL achieved a METRIC in a PERIOD
  COMPARED_TO  — METRIC compared to another METRIC across different PERIOD

Limit to the 10 most important entities and 10 most important relations.
If nothing is extractable, return empty arrays.

{_BASE_RULES}""",

    "report": f"""You are a named entity recognition system specialized in Philippine school accomplishment and narrative reports.
Extract entities and relationships from accomplishment reports, IPCRF, RPMS narratives, and program reports.

Valid entity types:
  TEACHER     — Named teachers who implemented or participated
  SCHOOL_HEAD — Named principal or head teacher
  OFFICIAL    — Named division or regional officials
  SCHOOL      — Named school
  DIVISION    — School division
  PROGRAM     — Named programs or projects e.g. "Brigada Eskwela", "Reading Program"
  METRIC      — Quantified outcomes e.g. "95% participation rate", "120 beneficiaries"
  PERIOD      — Reporting period, school year, quarter
  ASSESSMENT  — Named evaluations or monitoring activities

Valid relation types:
  IMPLEMENTED_BY  — PROGRAM implemented by TEACHER or SCHOOL_HEAD
  CONDUCTED_AT    — PROGRAM or ASSESSMENT conducted at SCHOOL
  REPORTED_IN     — METRIC reported in a PERIOD
  SUPERVISED_BY   — PROGRAM supervised by SCHOOL_HEAD or OFFICIAL
  PARTICIPATED_IN — TEACHER participated in a PROGRAM
  ACHIEVED        — SCHOOL or PROGRAM achieved a METRIC
  COVERED         — PROGRAM covered a PERIOD or beneficiary group
  ENDORSED_BY     — Report endorsed by SCHOOL_HEAD or OFFICIAL

Limit to the 10 most important entities and 10 most important relations.
If nothing is extractable, return empty arrays.

{_BASE_RULES}""",

    "content": f"""You are a named entity recognition system specialized in Philippine teaching and learning materials.
Extract entities and relationships from stories, songs, reading passages, e-learning modules, worksheets, and presentations.

Valid entity types:
  SUBJECT     — Learning area this material belongs to
  GRADE_LEVEL — Target grade level e.g. "Grade 2", "Kinder"
  COMPETENCY  — Learning objective or MELC this material addresses
  PROGRAM     — Teaching approach e.g. "Marungko", "MTB-MLE", "Big Books"
  CONCEPT     — Key idea or topic being taught e.g. "Pagmamahal sa Bayan", "Water Cycle"
  CHARACTER   — Named characters in stories or scenarios
  SETTING     — Named places or settings in the material
  TEACHER     — Named teacher if the material is attributed

Valid relation types:
  TEACHES_CONCEPT — material or SUBJECT teaches a CONCEPT
  SUITABLE_FOR    — material suitable for a GRADE_LEVEL
  ALIGNED_WITH    — material aligned with a COMPETENCY
  USES_APPROACH   — material uses a PROGRAM or teaching method
  FEATURES        — material features a CHARACTER or SETTING
  BELONGS_TO      — material belongs to a SUBJECT

Limit to 5 most important entities and 5 most important relations.
Keep extraction minimal — this profile is used for retrieval support, not deep graph traversal.
If nothing is extractable, return empty arrays.

{_BASE_RULES}""",
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
    profile = classify_document(file_name, metadata)
    system_prompt = _PROMPTS[profile]

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

    user_message = f"{context_hint}Text:\n{text_input}"

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