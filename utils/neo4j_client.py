"""
neo4j_client.py
================
All Neo4j Aura operations for the RAG knowledge graph.

Drop this file into:  utils/neo4j_client.py

Responsibilities:
  - Singleton async driver (neo4j.AsyncGraphDatabase)
  - Schema / constraint bootstrap  (run once on startup)
  - store_graph_data()   — called by pipelineDocument.py after embedding
  - graph_search()       — called by tools.py graph_search_tool
  - delete_graph_for_file() — called on file deletion / re-processing
"""

import os
import sys
import asyncio
import logging

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from neo4j import AsyncGraphDatabase, AsyncDriver
from neo4j.exceptions import ServiceUnavailable, SessionExpired, TransientError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

# Retry decorator for any Neo4j operation that may hit a dropped connection.
# Covers: ServiceUnavailable (routing gone), SessionExpired (conn recycled),
# TransientError (Aura transient fault), ConnectionResetError (TCP drop).
def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, (ServiceUnavailable, SessionExpired, TransientError)):
        return True
    # ConnectionResetError surfaces wrapped inside a generic Exception
    if isinstance(exc, Exception) and "defunct connection" in str(exc).lower():
        return True
    if isinstance(exc, ConnectionResetError):
        return True
    return False

_neo4j_retry = retry(
    retry=retry_if_exception_type((ServiceUnavailable, SessionExpired,
                                   TransientError, ConnectionResetError)),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)

log = logging.getLogger(__name__)

# ── Singleton async driver ────────────────────────────────────────────────────
_driver: AsyncDriver | None = None


def get_neo4j() -> AsyncDriver:
    global _driver
    if _driver is None:
        # neo4j+s:// encodes TLS in the scheme — do NOT pass encrypted=True.
        # Connection pool tuned for Aura Free which aggressively closes idle
        # connections after ~60s. Keep-alive + short liveness check prevents
        # "defunct connection" / ConnectionResetError mid-query.
        _driver = AsyncGraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
            # Pool settings
            max_connection_lifetime=60 * 3,      # recycle connections every 3 min
            max_connection_pool_size=10,
            connection_acquisition_timeout=30,
            # Liveness: test connection before handing it to a session
            liveness_check_timeout=10,
            # Keep-alive at TCP level
            keep_alive=True,
        )
    return _driver


async def close_neo4j():
    """Call this on app shutdown to cleanly close the driver."""
    global _driver
    if _driver:
        await _driver.close()
        _driver = None


# ── Schema bootstrap ──────────────────────────────────────────────────────────
# Idempotent — safe to call on every startup.
# Creates uniqueness constraints so MERGE never duplicates nodes.

SCHEMA_QUERIES = [
    # One node per (user_id, file_name) pair
    """CREATE CONSTRAINT document_unique IF NOT EXISTS
       FOR (d:Document) REQUIRE (d.user_id, d.file_name) IS UNIQUE""",

    # One Entity node per (user_id, name, type) — scoped per user
    """CREATE CONSTRAINT entity_unique IF NOT EXISTS
       FOR (e:Entity) REQUIRE (e.user_id, e.name, e.type) IS UNIQUE""",

    # Index for fast entity lookups by name within a user's graph
    """CREATE INDEX entity_name_idx IF NOT EXISTS
       FOR (e:Entity) ON (e.user_id, e.name)""",

    # Index for fast document lookups
    """CREATE INDEX document_user_idx IF NOT EXISTS
       FOR (d:Document) ON (d.user_id)""",
]


async def verify_connection(timeout: float = 30.0) -> bool:
    """
    Ping Neo4j via verify_connectivity() — the official driver health-check method.
    Logs the raw exception type+message so connection issues are always diagnosable.
    """
    driver = get_neo4j()
    try:
        await asyncio.wait_for(driver.verify_connectivity(), timeout=timeout)
        log.info("[neo4j] Connection verified ✓")
        return True
    except asyncio.TimeoutError:
        log.error(
            f"[neo4j] Connection timed out after {timeout}s.\n"
            f"  NEO4J_URI  = {NEO4J_URI!r}\n"
            f"  NEO4J_USER = {NEO4J_USER!r}\n"
            "  → Free instances pause after 3 days — resume at https://console.neo4j.io"
        )
        return False
    except Exception as e:
        log.error(
            f"[neo4j] Connection FAILED\n"
            f"  Exception  : {type(e).__name__}: {e}\n"
            f"  NEO4J_URI  = {NEO4J_URI!r}\n"
            f"  NEO4J_USER = {NEO4J_USER!r}\n"
            "  Checklist:\n"
            "  1. URI format: neo4j+s://<id>.databases.neo4j.io  (no port, no trailing slash)\n"
            "  2. Password correct? Aura only shows it once at creation\n"
            "  3. Instance status must be RUNNING (not Paused/Loading) in Aura console\n"
            "  4. Free instances auto-pause after 3 days idle"
        )
        return False


@_neo4j_retry
async def bootstrap_schema():
    """
    Run once on server startup (called from main.py lifespan).
    Creates constraints and indexes if they don't already exist.
    """
    driver = get_neo4j()
    async with driver.session() as session:
        for query in SCHEMA_QUERIES:
            try:
                await session.run(query)
            except Exception as e:
                # Some Aura tiers use slightly different syntax — log and continue
                log.warning(f"[neo4j] Schema query skipped ({e}): {query[:60]}...")
    log.info("[neo4j] Schema bootstrap complete")


# ── Write: store entities + relations for one chunk ──────────────────────────

@_neo4j_retry
async def store_graph_data(
    user_id:   str,
    file_name: str,
    job_id:    str,
    chunk_index: int,
    entities:  list[dict],   # [{"name": "Acme Corp", "type": "ORG"}, ...]
    relations: list[dict],   # [{"source": "Acme Corp", "target": "Manila", "type": "LOCATED_IN"}, ...]
) -> None:
    """
    Upsert Document → Entity nodes and APPEARS_IN edges, plus native
    relationship-type edges between entities (e.g. ISSUED_BY, COVERS).

    All nodes are scoped to user_id so users never see each other's data.
    Uses MERGE so re-processing the same file is safe (idempotent).
    Relationship types are created dynamically via apoc.merge.relationship().
    """
    if not entities and not relations:
        return

    driver = get_neo4j()
    chunk_ref = f"{job_id}::{chunk_index}"

    async with driver.session() as session:

        # 1. Upsert Document node
        await session.run("""
            MERGE (d:Document {user_id: $user_id, file_name: $file_name})
            SET d.job_id     = $job_id,
                d.updated_at = timestamp()
        """, user_id=user_id, file_name=file_name, job_id=job_id)

        # 2. Upsert Entity nodes + link each to Document
        for entity in entities:
            name = (entity.get("name") or "").strip()
            etype = (entity.get("type") or "UNKNOWN").strip().upper()
            if not name:
                continue

            await session.run("""
                MERGE (e:Entity {user_id: $user_id, name: $name, type: $etype})
                SET e.updated_at = timestamp()
                WITH e
                MATCH (d:Document {user_id: $user_id, file_name: $file_name})
                MERGE (e)-[r:APPEARS_IN]->(d)
                SET r.chunks = CASE
                    WHEN $chunk_ref IN coalesce(r.chunks, [])
                    THEN r.chunks
                    ELSE coalesce(r.chunks, []) + $chunk_ref
                END
            """, user_id=user_id, name=name, etype=etype,
                 file_name=file_name, chunk_ref=chunk_ref)

        # 3. Upsert native relationship edges between entity pairs
        #    Uses apoc.merge.relationship() so each relation type
        #    (ISSUED_BY, COVERS, etc.) becomes its own Neo4j edge type.
        for rel in relations:
            source = (rel.get("source") or "").strip()
            target = (rel.get("target") or "").strip()
            rel_type = (rel.get("type") or "RELATED_TO").strip().upper().replace(" ", "_")
            if not source or not target:
                continue

            await session.run("""
                MATCH (a:Entity {user_id: $user_id, name: $source})
                MATCH (b:Entity {user_id: $user_id, name: $target})
                CALL apoc.merge.relationship(a, $rel_type, {}, {updated_at: timestamp(), chunk_ref: $chunk_ref}, b) YIELD rel
                RETURN rel
            """, user_id=user_id, source=source, target=target,
                 rel_type=rel_type, chunk_ref=chunk_ref)


# ── Read: search graph by entity names extracted from the query ───────────────

@_neo4j_retry
async def graph_search(
    user_id:        str,
    entity_names:   list[str],
    max_hops:       int = 2,
    limit:          int = 10,
) -> list[dict]:
    """
    Find documents and related entities connected to the queried entity names.

    Two-hop traversal using any relationship type except APPEARS_IN:
      (query_entity) -[*1..max_hops]-> (neighbor) -[:APPEARS_IN]-> (doc)

    Returns list of:
      {
        "file_name":        "report.pdf",
        "matched_entities": ["Acme Corp", "Manila"],
        "related_entities": ["John Doe", "2024"],
        "relevance_score":  3
      }
    """
    if not entity_names:
        return []

    driver = get_neo4j()

    async with driver.session() as session:
        result = await session.run("""
            MATCH (e:Entity {user_id: $user_id})
            WHERE e.name IN $names
            MATCH (e)-[:APPEARS_IN]->(d:Document {user_id: $user_id})

            OPTIONAL MATCH path = (e)-[*1..2]-(neighbor:Entity {user_id: $user_id})
            WHERE NONE(r IN relationships(path) WHERE type(r) = 'APPEARS_IN')
            OPTIONAL MATCH (neighbor)-[:APPEARS_IN]->(d2:Document {user_id: $user_id})

            WITH d,
                 collect(DISTINCT e.name)        AS matched_entities,
                 collect(DISTINCT neighbor.name)  AS related_entities,
                 count(DISTINCT e)                AS relevance_score

            RETURN d.file_name        AS file_name,
                   matched_entities,
                   related_entities,
                   relevance_score
            ORDER BY relevance_score DESC
            LIMIT $limit
        """, user_id=user_id, names=entity_names, limit=limit)

        rows = await result.data()
        return rows or []


@_neo4j_retry
async def get_entity_neighbors(
    user_id:     str,
    entity_name: str,
    limit:       int = 20,
) -> list[dict]:
    """
    Return all entities directly connected to a named entity.
    Useful for graph exploration / "tell me more about X" queries.
    """
    driver = get_neo4j()
    async with driver.session() as session:
        result = await session.run("""
            MATCH (a:Entity {user_id: $user_id, name: $name})
            MATCH (a)-[r]-(b:Entity {user_id: $user_id})
            WHERE type(r) <> 'APPEARS_IN'
            RETURN b.name AS neighbor, b.type AS type, type(r) AS relation
            LIMIT $limit
        """, user_id=user_id, name=entity_name, limit=limit)
        return await result.data()


# ── Delete: clean up when a file is re-processed or deleted ──────────────────

async def delete_graph_for_file(user_id: str, file_name: str) -> int:
    """
    Remove the Document node and all APPEARS_IN edges for a file.
    Orphaned Entity nodes (not appearing in any other document) are also removed.

    Returns the number of nodes deleted.
    """
    driver = get_neo4j()
    async with driver.session() as session:
        # Detach-delete the Document and its APPEARS_IN edges
        result = await session.run("""
            MATCH (d:Document {user_id: $user_id, file_name: $file_name})
            DETACH DELETE d
        """, user_id=user_id, file_name=file_name)
        summary = await result.consume()

        # Clean up orphan entities (no remaining APPEARS_IN edges)
        # DETACH DELETE removes the entity AND any entity-to-entity edges
        # (e.g. ISSUED_BY, COVERS) still attached to the orphan.
        orphan_result = await session.run("""
            MATCH (e:Entity {user_id: $user_id})
            WHERE NOT (e)-[:APPEARS_IN]->()
            DETACH DELETE e
            RETURN count(e) AS deleted
        """, user_id=user_id)
        orphan_data = await orphan_result.data()
        orphan_count = orphan_data[0]["deleted"] if orphan_data else 0

        deleted = summary.counters.nodes_deleted + orphan_count
        log.info(f"[neo4j] Deleted graph for {file_name}: {deleted} nodes removed")
        return deleted