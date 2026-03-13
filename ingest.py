import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from fastapi import APIRouter, Request, HTTPException, BackgroundTasks, Depends
from auth import verify_webhook_secret
from pipeline import process_file_pipeline
from config import STORAGE_BUCKET

router = APIRouter()


@router.post("/webhook/file-uploaded")
async def file_uploaded(
    request:          Request,
    background_tasks: BackgroundTasks,
):
    """
    Receives webhook from Supabase Postgres trigger when a row
    is inserted into user_files.

    Payload shape (matches trigger jsonb_build_object):
    {
      "user_file_id":   "uuid",
      "user_id":        "uuid",
      "storage_path":   "path/to/file.pdf",
      "file_name":      "report.pdf",
      "mime_type":      "application/pdf",
      "file_size":      102400,
      "storage_bucket": "user-files"
    }

    Returns 200 immediately. Processing runs in background.
    """
    # Verify secret before reading body
    await verify_webhook_secret(request)

    payload = await request.json()

    # Validate required fields
    required = ["user_file_id", "user_id", "storage_path", "file_name", "mime_type"]
    missing  = [f for f in required if not payload.get(f)]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing required fields: {missing}"
        )

    bucket = payload.get("storage_bucket", STORAGE_BUCKET)

    # Queue background task — do NOT await this
    background_tasks.add_task(
        process_file_pipeline,
        user_file_id=payload["user_file_id"],
        user_id=payload["user_id"],
        storage_path=payload["storage_path"],
        file_name=payload["file_name"],
        mime_type=payload["mime_type"],
        bucket=bucket,
    )

    # Return immediately so Postgres trigger doesn't wait or timeout
    return {
        "status":        "queued",
        "user_file_id":  payload["user_file_id"],
        "file_name":     payload["file_name"],
    }
