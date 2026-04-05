"""
entity_extractor.py
====================
Extracts named entities and their relationships from text chunks
using GPT-4o-mini with structured JSON output.

Drop this file into:  services/entity_extractor.py

Called by pipelineDocument.py once per chunk, right after embedding.
Returns:
  {
    "entities":  [{"name": "Acme Corp",  "type": "ORG"}, ...],
    "relations": [{"source": "Acme Corp", "target": "Manila", "type": "LOCATED_IN"}, ...]
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
from config import OPENAI_API_KEY

log = logging.getLogger(__name__)

_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# ── Entity types the extractor understands ─────────────────────────────────────
# Tune this list to your domain.  Common additions:
#   PRODUCT, METRIC, REGULATION, DEPARTMENT, TECHNOLOGY
ENTITY_TYPES = [
    "PERSON",       # Named individuals
    "ORG",          # Companies, institutions, agencies
    "LOCATION",     # Cities, countries, regions, addresses
    "DATE",         # Specific dates, periods, fiscal years
    "CONCEPT",      # Abstract ideas, topics, themes
    "PRODUCT",      # Software, hardware, branded offerings
    "METRIC",       # KPIs, figures, percentages, quantities
    "TECHNOLOGY",   # Programming languages, frameworks, tools
]

SYSTEM_PROMPT = f"""You are a precise named entity recognition (NER) system.

Given a text chunk, extract:
1. Named entities — specific, unambiguous names (not generic words like "company" or "document")
2. Relationships between those entities — only when clearly stated in the text

Valid entity types: {", ".join(ENTITY_TYPES)}

Rules:
- Extract ONLY entities explicitly named in the text
- Entity names must be the canonical form (e.g. "United States" not "the US")
- Skip pronouns, generic nouns, and vague references
- Relationships must have a clear, specific type (e.g. WORKS_FOR, LOCATED_IN, ACQUIRED, REPORTS_TO)
- Limit to the 10 most important entities and 10 most important relations
- If nothing is extractable, return empty arrays

Respond with ONLY valid JSON — no markdown fences, no explanation:
{{
  "entities":  [{{"name": "string", "type": "ENTITY_TYPE"}}, ...],
  "relations": [{{"source": "entity name", "target": "entity name", "type": "RELATION_TYPE"}}, ...]
}}"""


async def extract_entities_and_relations(chunk_text: str) -> dict:
    """
    Run NER + relation extraction on a single text chunk.

    Uses response_format=json_object for guaranteed parseable output.
    Falls back to empty result on any error so the main pipeline never fails
    because of a graph extraction issue.

    Args:
        chunk_text: The raw text of one chunk (ideally ≤ 1500 chars)

    Returns:
        {"entities": [...], "relations": [...]}
    """
    if not chunk_text or not chunk_text.strip():
        return {"entities": [], "relations": []}

    # Truncate to avoid token waste on very large chunks
    text_input = chunk_text[:2000].strip()

    try:
        response = await _client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": text_input},
            ],
            temperature=0.0,       # deterministic extraction
            max_tokens=600,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content.strip()
        parsed = json.loads(raw)

        entities  = _validate_entities(parsed.get("entities", []))
        relations = _validate_relations(parsed.get("relations", []), entities)

        log.debug(
            f"[entity_extractor] extracted {len(entities)} entities, "
            f"{len(relations)} relations from chunk ({len(text_input)} chars)"
        )
        return {"entities": entities, "relations": relations}

    except json.JSONDecodeError as e:
        log.warning(f"[entity_extractor] JSON parse failed: {e}")
        return {"entities": [], "relations": []}

    except Exception as e:
        log.warning(f"[entity_extractor] Extraction failed: {e}")
        return {"entities": [], "relations": []}


# ── Validation helpers ─────────────────────────────────────────────────────────

def _validate_entities(raw: list) -> list[dict]:
    """
    Sanitize entity list — remove malformed entries and normalize types.
    """
    valid = []
    seen = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        etype = str(item.get("type", "CONCEPT")).strip().upper()

        # Skip empty names, single-char names, pure numbers
        if not name or len(name) < 2 or name.isdigit():
            continue

        # Normalize type to our accepted list
        if etype not in ENTITY_TYPES:
            etype = "CONCEPT"

        # Deduplicate within this chunk
        key = (name.lower(), etype)
        if key in seen:
            continue
        seen.add(key)

        valid.append({"name": name, "type": etype})

    return valid[:10]   # cap at 10 per chunk


def _validate_relations(raw: list, entities: list[dict]) -> list[dict]:
    """
    Sanitize relation list — both endpoints must exist in the extracted entities.
    """
    entity_names = {e["name"].lower() for e in entities}
    valid = []
    seen = set()

    for item in raw:
        if not isinstance(item, dict):
            continue
        source   = str(item.get("source", "")).strip()
        target   = str(item.get("target", "")).strip()
        rel_type = str(item.get("type",   "RELATED_TO")).strip().upper().replace(" ", "_")

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

    return valid[:10]   # cap at 10 per chunk


# ── Batch helper (for future use / testing) ───────────────────────────────────

async def extract_batch(chunks: list[str]) -> list[dict]:
    """
    Extract entities from multiple chunks concurrently.
    Limits concurrency to 5 to avoid rate-limit spikes.
    """
    import asyncio
    semaphore = asyncio.Semaphore(5)

    async def _extract_one(text: str) -> dict:
        async with semaphore:
            return await extract_entities_and_relations(text)

    return await asyncio.gather(*[_extract_one(c) for c in chunks])
