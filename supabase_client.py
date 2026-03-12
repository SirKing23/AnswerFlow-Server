from supabase import create_client, Client
from config import SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY

# Service role client — can read any user's files from storage
# Never expose this key to the frontend
_client: Client = None

def get_supabase() -> Client:
    global _client
    if _client is None:
        _client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    return _client


async def download_file_from_storage(storage_path: str, bucket: str) -> bytes:
    """Download a file from Supabase Storage and return raw bytes."""
    supabase = get_supabase()
    response = supabase.storage.from_(bucket).download(storage_path)
    return response


async def create_job(user_file_id: str, user_id: str, storage_path: str,
                     file_name: str, file_type: str) -> str:
    """Insert a new processing job and return its id."""
    supabase = get_supabase()
    result = supabase.table("file_processing_jobs").insert({
        "user_file_id": user_file_id,
        "user_id":      user_id,
        "storage_path": storage_path,
        "file_name":    file_name,
        "file_type":    file_type,
        "status":       "queued",
    }).execute()
    return result.data[0]["id"]


async def update_job_status(job_id: str, status: str,
                            error: str = None, chunk_count: int = None):
    """Update file_processing_jobs status."""
    supabase = get_supabase()
    payload = {"status": status, "updated_at": "now()"}
    if error:
        payload["error_message"] = error
    if chunk_count is not None:
        payload["chunk_count"] = chunk_count
    supabase.table("file_processing_jobs").update(payload).eq("id", job_id).execute()


async def update_file_status(user_file_id: str, processing_status: str,
                             job_id: str = None, processed_at: bool = False,
                             content_hash: str = None):
    """Update user_files processing_status column."""
    supabase = get_supabase()
    payload = {
        "processing_status": processing_status,
        "updated_at":        "now()",
    }
    if job_id:
        payload["processing_job_id"] = job_id
    if processed_at:
        from datetime import datetime, timezone
        payload["processed_at"] = datetime.now(timezone.utc).isoformat()
    if content_hash:
        payload["content_hash"] = content_hash

    supabase.table("user_files").update(payload).eq("id", user_file_id).execute()


async def check_duplicate(content_hash: str, user_id: str) -> bool:
    """Return True if this exact file content was already embedded for this user."""
    supabase = get_supabase()
    result = (
        supabase.table("user_files")
        .select("id")
        .eq("user_id", user_id)
        .eq("content_hash", content_hash)
        .eq("processing_status", "done")
        .limit(1)
        .execute()
    )
    return len(result.data) > 0


async def store_embeddings(rows: list[dict]):
    """Bulk insert embedding rows into document_embeddings."""
    supabase = get_supabase()
    supabase.table("document_embeddings").insert(rows).execute()
