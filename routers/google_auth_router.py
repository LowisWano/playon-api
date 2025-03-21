from fastapi import APIRouter, Depends, HTTPException, status
from services.google_auth_services import GoogleAuthService
from middleware.auth_middleware import get_current_user
from prisma.models import Users
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/auth/google",
    tags=["auth"]
)

google_auth_service = GoogleAuthService()

@router.get("/login")
async def login_google():
    """Get Google OAuth login URL"""
    logger.debug("Login endpoint hit")
    return {"url": google_auth_service.get_auth_url()}

@router.get("/callback")
async def auth_google(code: str):
    """Handle Google OAuth callback and user authentication"""
    logger.debug("Callback endpoint hit")
    try:
        # Get tokens from Google
        tokens = await google_auth_service.get_tokens(code)
        access_token = tokens.get("access_token")
        
        if not access_token:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to get access token from Google"
            )
            
        # Get user info from Google
        user_info = await google_auth_service.get_user_info(access_token)
        
        if not user_info.get("email"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to get user email from Google"
            )
            
        # Authenticate/create user
        result = await google_auth_service.authenticate_google_user(user_info)
        
        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result["error"]
            )
            
        return {"access_token": result["token"], "token_type": "bearer"}
        
    except Exception as e:
        logger.error(f"Error in callback: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/me", response_model=dict)
async def get_current_user_info(current_user: Users = Depends(get_current_user)):
    """Get current authenticated user info"""
    logger.debug("Me endpoint hit")
    return {
        "id": current_user.id,
        "email": current_user.email,
        "first_name": current_user.first_name,
        "last_name": current_user.last_name,
        "profile_pic": current_user.profile_pic,
        "is_verified": current_user.is_verified
    }