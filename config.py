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

# Unstructured.io
UNSTRUCTURED_API_KEY      = os.environ["UNSTRUCTURED_API_KEY"]
UNSTRUCTURED_API_URL      = os.getenv(
    "UNSTRUCTURED_API_URL",
    "https://api.unstructuredapp.io/general/v0/general"
)

# Security
WEBHOOK_SECRET            = os.environ["WEBHOOK_SECRET"]

# Processing config
MAX_FILE_SIZE_BYTES       = int(os.getenv("MAX_FILE_SIZE_MB", 25)) * 1024 * 1024
CHUNK_SIZE_TOKENS         = int(os.getenv("CHUNK_SIZE_TOKENS", 512))
CHUNK_OVERLAP_TOKENS      = int(os.getenv("CHUNK_OVERLAP_TOKENS", 64))
EMBEDDING_BATCH_SIZE      = int(os.getenv("EMBEDDING_BATCH_SIZE", 100))

# Storage bucket — must match your Supabase bucket name
STORAGE_BUCKET            = os.getenv("STORAGE_BUCKET", "user-files")

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
