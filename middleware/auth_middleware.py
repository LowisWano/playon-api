from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
import os
from dotenv import load_dotenv
from db.prisma import prisma  # Use the singleton instance instead

load_dotenv()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, os.getenv("JWT_SECRET_KEY"), algorithms=["HS256"])
        user_id = int(payload.get("sub"))
        if user_id is None:
            raise credentials_exception
        
        user = await prisma.users.find_unique(where={"id": user_id})
        
        if user is None:
            raise credentials_exception
            
        return user
        
    except JWTError:
        raise credentials_exception