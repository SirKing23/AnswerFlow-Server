import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
