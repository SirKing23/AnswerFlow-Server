import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jwt
from fastapi import Request, HTTPException
from config import WEBHOOK_SECRET, SUPABASE_JWT_SECRET


async def verify_webhook_secret(request: Request) -> None:
    """
    Verify that the incoming webhook request came from our Supabase trigger.
    Raises 401 if the secret header is missing or wrong.
    """
    secret = request.headers.get("X-Webhook-Secret")
    if not secret or secret != WEBHOOK_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized webhook call")


async def verify_jwt(request: Request) -> str:
    """
    Verify Supabase JWT from Authorization header.
    Returns the user_id (sub claim) if valid.
    Raises 401 if token is missing, expired, or invalid.

    Frontend must send:
      Authorization: Bearer <supabase_access_token>
    """
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    token = auth_header.split(" ")[1]

    try:
        payload = jwt.decode(
            token,
            SUPABASE_JWT_SECRET,
            algorithms=["HS256"],
            options={"verify_aud": False}  # Supabase tokens don't use audience
        )
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token — no user id")
        return user_id

    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")
