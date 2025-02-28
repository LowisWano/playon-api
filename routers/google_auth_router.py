from fastapi import APIRouter, Depends
from services.google_auth_services import GoogleAuthService
from middleware.auth_middleware import get_current_user

router = APIRouter(
    prefix="/auth/google",
    tags=["auth"]
)

google_auth_service = GoogleAuthService()

@router.get("/login")
async def login_google():
    return {"url": google_auth_service.get_auth_url()}

@router.get("/callback")
async def auth_google(code: str):
    tokens = await google_auth_service.get_tokens(code)
    access_token = tokens.get("access_token")
    if access_token:
        user_info = await google_auth_service.get_user_info(access_token)
        return user_info
    return {"error": "Failed to get access token"}

@router.get("/token")
async def get_token(token: str = Depends(get_current_user)):
    return google_auth_service.decode_token(token)