"""
entity_extractor.py
====================
Extracts named entities and their relationships from text chunks
using GPT-4o-mini with structured JSON output.

Drop this file into:  services/entity_extractor.py

Documentation-domain aware — classifies each document into one of 5 profiles
and uses a specialized extraction prompt per profile for higher accuracy.

Profiles:
  governance  — Policies, SOPs, compliance docs, security guidelines, governance docs
  guide       — User guides, admin guides, installation guides, tutorials, setup docs
  reference   — API references, specs, data dictionaries, SDK docs, OpenAPI
  report      — Release notes, changelogs, incident reports, audit reports, post-mortems
  general     — README, architecture overviews, whitepapers, FAQs, presentations

Called by pipelineDocument.py once per chunk, right after embedding.

Signature change from the original:
  extract_entities_and_relations(chunk_text, file_name, metadata) → dict

Returns:
  {
    "entities":  [{"name": "PostgreSQL", "type": "PRODUCT"}, ...],
    "relations": [{"source": "PostgreSQL", "target": "v14.2",
                   "type": "HAS_VERSION"}, ...]
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
    # People & Organizations
    "PERSON",        # Named individuals: authors, approvers, contacts
    "ORGANIZATION",  # Companies, teams, departments, vendors
    "ROLE",          # User roles, permissions, job titles e.g. "Administrator", "Developer"

    # Products & Components
    "PRODUCT",       # Named software, tools, services, platforms e.g. "Docker", "PostgreSQL"
    "COMPONENT",     # Modules, services, APIs, libraries, plugins, packages
    "VERSION",       # Software versions, releases e.g. "v2.1.0", "Release 3", "API v4"
    "FEATURE",       # Named features, capabilities, functionality flags

    # API & Interface
    "ENDPOINT",      # API endpoints, routes, URLs e.g. "/api/v1/users", "GET /documents"
    "PARAMETER",     # Config params, env variables, flags, settings keys, request fields
    "ERROR_CODE",    # Error codes, HTTP status codes, exception types e.g. "404", "AuthException"

    # Technical Context
    "REQUIREMENT",   # Prerequisites, dependencies, system requirements
    "PLATFORM",      # OS, cloud environments, infrastructure e.g. "Linux", "AWS", "Docker"
    "STANDARD",      # Protocols, specs, formats e.g. "OAuth 2.0", "REST", "OpenAPI 3.0"

    # Governance
    "POLICY",        # Security policies, compliance rules, governance documents
    "PROCESS",       # Named procedures, workflows, SOPs

    # Measurement & Time
    "METRIC",        # Performance metrics, KPIs, thresholds, limits
    "PERIOD",        # Dates, timelines, deadlines, version release windows

    # General
    "CONCEPT",       # Technical concepts, terminology, abstract ideas
]


# ── Document profile classifier ────────────────────────────────────────────────

# Filename keyword signals per profile — checked first (fast, no LLM needed)
_FILENAME_SIGNALS: dict[str, list[str]] = {
    "governance": [
        "policy", "compliance", "sop", "standard operating procedure",
        "governance", "regulation", "regulatory", "bylaw", "by-law",
        "security policy", "access control", "data protection", "gdpr",
        "acceptable use", "terms", "code of conduct", "audit", "controls",
        "information security", "privacy", "risk",
    ],
    "guide": [
        "guide", "tutorial", "how-to", "howto", "installation", "install",
        "setup", "configuration", "getting started", "quickstart", "quick start",
        "walkthrough", "manual", "handbook", "user guide", "admin guide",
        "administrator guide", "developer guide", "deployment guide",
        "migration guide", "upgrade guide", "onboarding", "troubleshoot",
    ],
    "reference": [
        "api", "reference", "spec", "specification", "schema",
        "data dictionary", "parameter", "endpoint", "swagger", "openapi",
        "sdk", "interface", "glossary", "dictionary", "readme",
    ],
    "report": [
        "release note", "release notes", "changelog", "change log",
        "incident", "postmortem", "post-mortem", "post mortem",
        "status report", "runbook", "audit report", "assessment report",
        "findings", "summary report",
    ],
    "general": [
        "overview", "introduction", "architecture", "design",
        "whitepaper", "white paper", "presentation", "concept",
        "about", "faq", "background",
    ],
}

# element_types composition signals — used when filename is ambiguous
_ELEMENT_TYPE_SIGNALS: dict[str, list[str]] = {
    "reference": ["table"],
    "governance": ["heading"],
    "general":    ["narrative_text", "list_item"],
}


def classify_document(file_name: str, metadata: dict) -> str:
    """
    Classify a document into one of 5 profiles without an LLM call.

    Priority:
      1. Filename keyword match  (most reliable)
      2. element_types composition from metadata
      3. Default to 'general'   (safest fallback for unrecognized docs)

    Args:
        file_name: Original filename e.g. "API Reference v3.pdf"
        metadata:  Chunk metadata dict with optional keys:
                     element_types, headings, section

    Returns:
        One of: "governance", "guide", "reference", "report", "general"
    """
    name_lower = file_name.lower()

    # 1. Filename signals — check each profile in priority order
    priority_order = ["governance", "reference", "guide", "report", "general"]
    for profile in priority_order:
        if any(signal in name_lower for signal in _FILENAME_SIGNALS[profile]):
            log.debug(f"[entity_extractor] '{file_name}' → profile={profile} (filename match)")
            return profile

    # 2. element_types composition
    element_types = metadata.get("element_types", [])
    if element_types:
        type_set = set(element_types)
        if "table" in type_set:
            log.debug(f"[entity_extractor] '{file_name}' → profile=reference (table elements)")
            return "reference"
        # Heading-heavy without table → likely governance or guide
        heading_count = element_types.count("heading")
        text_count    = element_types.count("text") + element_types.count("narrative_text")
        if heading_count > 2 and text_count > heading_count:
            log.debug(f"[entity_extractor] '{file_name}' → profile=governance (heading-heavy)")
            return "governance"

    # 3. Default
    log.debug(f"[entity_extractor] '{file_name}' → profile=general (default fallback)")
    return "general"


# ── Extraction prompts per profile ─────────────────────────────────────────────

# Prompts loaded from .env via config.py.
# ENTITY_BASE_RULES (shared JSON rules) is injected into each profile via
# the <<BASE_RULES>> placeholder, which config.py replaces at startup.
# Edit the ENTITY_PROMPT_* variables in .env to add entity/relation types without code changes.
_PROMPTS: dict[str, str] = {
    "governance": ENTITY_PROMPT_POLICY,       # governance, compliance, SOP docs
    "guide":      ENTITY_PROMPT_CURRICULUM,   # user/admin/developer guides, tutorials
    "reference":  ENTITY_PROMPT_PERFORMANCE,  # API refs, specs, data dictionaries
    "report":     ENTITY_PROMPT_REPORT,       # release notes, changelogs, incidents
    "general":    ENTITY_PROMPT_CONTENT,      # README, overviews, whitepapers
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
    # Prompt: _PROMPTS[profile] — one of governance/guide/reference/report/general,
    # each loaded from .env via config.py (ENTITY_BASE_RULES injected)
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

        choice = response.choices[0]
        if choice.finish_reason == "length":
            log.warning(
                f"[entity_extractor] Response truncated (max_tokens hit) for '{file_name}' — skipping chunk"
            )
            return {"entities": [], "relations": []}

        raw    = choice.message.content.strip()
        parsed = json.loads(raw)

        # Determine entity cap based on profile
        entity_cap   = ENTITY_CAP_CONTENT if profile == "general" else ENTITY_CAP_DEFAULT
        relation_cap = RELATION_CAP_CONTENT if profile == "general" else RELATION_CAP_DEFAULT

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