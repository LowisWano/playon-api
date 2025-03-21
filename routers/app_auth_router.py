from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel, EmailStr
from datetime import datetime
from typing import Optional
from services.app_auth_service import AppAuthService
from middleware.auth_middleware import get_current_user
from prisma.models import Users
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/auth/app",
    tags=["auth"]
)

app_auth_service = AppAuthService()

class UserSignUp(BaseModel):
    email: EmailStr
    first_name: str
    last_name: str
    address: str
    birth_date: datetime
    gender: str
    password: str
    confirm_password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

@router.post("/signup")
async def signup(user_data: UserSignUp):
    """Handle user signup with app credentials"""
    logger.debug("Signup endpoint hit")
    try:
        # Validate passwords match
        if user_data.password != user_data.confirm_password:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Passwords do not match"
            )
            
        # Create user
        result = await app_auth_service.create_user(user_data)
        
        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result["error"]
            )
            
        return {"access_token": result["token"], "token_type": "bearer"}
        
    except Exception as e:
        logger.error(f"Error in signup: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/login")
async def login(user_data: UserLogin):
    """Handle user login with app credentials"""
    logger.debug("Login endpoint hit")
    try:
        result = await app_auth_service.authenticate_user(user_data)
        
        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=result["error"]
            )
            
        return {"access_token": result["token"], "token_type": "bearer"}
        
    except Exception as e:
        logger.error(f"Error in login: {str(e)}")
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
        "address": current_user.location,
        "birth_date": current_user.birth_date,
        "gender": current_user.gender,
        "profile_pic": current_user.profile_pic,
        "is_verified": current_user.is_verified
    } 