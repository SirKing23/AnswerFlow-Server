import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from fastapi import Request, HTTPException
from config import WEBHOOK_SECRET


async def verify_webhook_secret(request: Request) -> None:
    """
    Verify that the incoming webhook request came from our Supabase trigger.
    Raises 401 if the secret header is missing or wrong.
    """
    secret = request.headers.get("X-Webhook-Secret")
    if not secret or secret != WEBHOOK_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized webhook call")
