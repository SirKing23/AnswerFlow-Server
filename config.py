import os
from dotenv import load_dotenv

load_dotenv()

# Supabase
SUPABASE_URL              = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
SUPABASE_JWT_SECRET       = os.environ["SUPABASE_JWT_SECRET"]

# OpenAI
OPENAI_API_KEY            = os.environ["OPENAI_API_KEY"]
OPENAI_EMBEDDING_MODEL    = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_CHAT_MODEL         = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

# Unstructured.io
UNSTRUCTURED_API_KEY      = os.environ["UNSTRUCTURED_API_KEY"]
UNSTRUCTURED_API_URL      = os.getenv(
    "UNSTRUCTURED_API_URL",
    "https://api.unstructuredapp.io/general/v0/general"
)

# Neo4j Aura
# NEO4J_URI format: neo4j+s://<instance-id>.databases.neo4j.io
NEO4J_URI                 = os.environ["NEO4J_URI"]
NEO4J_USER                = os.getenv("NEO4J_USER", "none")
NEO4J_PASSWORD            = os.environ["NEO4J_PASSWORD"]

# Security
WEBHOOK_SECRET            = os.environ["WEBHOOK_SECRET"]

# Processing config
MAX_FILE_SIZE_BYTES       = int(os.getenv("MAX_FILE_SIZE_MB", 25)) * 1024 * 1024
CHUNK_SIZE_TOKENS         = int(os.getenv("CHUNK_SIZE_TOKENS", 512))
CHUNK_OVERLAP_TOKENS      = int(os.getenv("CHUNK_OVERLAP_TOKENS", 64))
EMBEDDING_BATCH_SIZE      = int(os.getenv("EMBEDDING_BATCH_SIZE", 100))

# Storage bucket — must match your Supabase bucket name
STORAGE_BUCKET            = os.getenv("STORAGE_BUCKET", "user-files")

# ── LLM Settings ──────────────────────────────────────────────────────────────
# Orchestrator agent (ai/orchestrator.py → run_agent)
ORCHESTRATOR_TEMPERATURE      = float(os.getenv("ORCHESTRATOR_TEMPERATURE", 0.7))
ORCHESTRATOR_PRESENCE_PENALTY = float(os.getenv("ORCHESTRATOR_PRESENCE_PENALTY", 0.3))
ORCHESTRATOR_MAX_TOKENS       = int(os.getenv("ORCHESTRATOR_MAX_TOKENS", 2048))
ORCHESTRATOR_MAX_TURNS        = int(os.getenv("ORCHESTRATOR_MAX_TURNS", 10))

# Query decomposer (ai/tools.py → query_decomposer_tool)
DECOMPOSER_TEMPERATURE = float(os.getenv("DECOMPOSER_TEMPERATURE", 0.1))
DECOMPOSER_MAX_TOKENS  = int(os.getenv("DECOMPOSER_MAX_TOKENS", 300))

# Context builder (ai/tools.py → context_builder_tool)
CONTEXT_BUILDER_TEMPERATURE = float(os.getenv("CONTEXT_BUILDER_TEMPERATURE", 0.1))
CONTEXT_BUILDER_MAX_TOKENS  = int(os.getenv("CONTEXT_BUILDER_MAX_TOKENS", 1500))

# Answer validator (ai/tools.py → answer_validator_tool)
VALIDATOR_TEMPERATURE = float(os.getenv("VALIDATOR_TEMPERATURE", 0.0))
VALIDATOR_MAX_TOKENS  = int(os.getenv("VALIDATOR_MAX_TOKENS", 400))

# Query NER for graph search (ai/tools.py → _extract_query_entities)
QUERY_NER_TEMPERATURE = float(os.getenv("QUERY_NER_TEMPERATURE", 0.0))
QUERY_NER_MAX_TOKENS  = int(os.getenv("QUERY_NER_MAX_TOKENS", 150))

# Text-to-Cypher generator (ai/tools.py → text2cypher_tool)
TEXT2CYPHER_TEMPERATURE = float(os.getenv("TEXT2CYPHER_TEMPERATURE", 0.0))
TEXT2CYPHER_MAX_TOKENS  = int(os.getenv("TEXT2CYPHER_MAX_TOKENS", 400))

# Simple RAG chat pipeline (services/pipelineChat.py → chat_with_docs)
CHAT_TEMPERATURE = float(os.getenv("CHAT_TEMPERATURE", 0.3))
CHAT_MAX_TOKENS  = int(os.getenv("CHAT_MAX_TOKENS", 1024))

# Entity extraction for knowledge graph (services/entity_extractor.py)
ENTITY_EXTRACTOR_TEMPERATURE = float(os.getenv("ENTITY_EXTRACTOR_TEMPERATURE", 0.0))
ENTITY_EXTRACTOR_MAX_TOKENS  = int(os.getenv("ENTITY_EXTRACTOR_MAX_TOKENS", 700))

# ── Search & Retrieval Settings ───────────────────────────────────────────────
# Vector search defaults (ai/tools.py, services/pipelineChat.py)
SEARCH_TOP_K                = int(os.getenv("SEARCH_TOP_K", 5))
SEARCH_SIMILARITY_THRESHOLD = float(os.getenv("SEARCH_SIMILARITY_THRESHOLD", 0.30))

# Knowledge graph settings
GRAPH_CONCURRENCY        = int(os.getenv("GRAPH_CONCURRENCY", 3))
GRAPH_SEARCH_LIMIT       = int(os.getenv("GRAPH_SEARCH_LIMIT", 8))
ENTITY_EXPLORER_LIMIT    = int(os.getenv("ENTITY_EXPLORER_LIMIT", 15))
TEXT2CYPHER_DEFAULT_LIMIT = int(os.getenv("TEXT2CYPHER_DEFAULT_LIMIT", 20))
TEXT2CYPHER_MAX_LIMIT     = int(os.getenv("TEXT2CYPHER_MAX_LIMIT", 50))

# Entity extraction caps (services/entity_extractor.py)
ENTITY_CAP_DEFAULT         = int(os.getenv("ENTITY_CAP_DEFAULT", 10))
ENTITY_CAP_CONTENT         = int(os.getenv("ENTITY_CAP_CONTENT", 5))
RELATION_CAP_DEFAULT       = int(os.getenv("RELATION_CAP_DEFAULT", 10))
RELATION_CAP_CONTENT       = int(os.getenv("RELATION_CAP_CONTENT", 5))
ENTITY_EXTRACT_CHAR_LIMIT  = int(os.getenv("ENTITY_EXTRACT_CHAR_LIMIT", 2000))

# Chat history window (ai/orchestrator.py, ai/tools.py → context_builder_tool)
CHAT_HISTORY_WINDOW = int(os.getenv("CHAT_HISTORY_WINDOW", 6))

# MIME types we handle ourselves (no Unstructured needed)
SELF_PARSE_MIME_TYPES = {
    "text/plain",
    "text/markdown",
    "text/csv",
    "text/tsv",
    "text/tab-separated-values",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # docx
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",        # xlsx
    "application/vnd.ms-excel",                                                  # xls
}

# MIME types that need Unstructured.io
UNSTRUCTURED_MIME_TYPES = {
    "application/pdf",
    "application/msword",                                                         # old .doc
    "application/vnd.ms-powerpoint",                                              # old .ppt
    "application/vnd.openxmlformats-officedocument.presentationml.presentation", # pptx
}
