from datetime import datetime, timedelta
from typing import Dict, Any
from prisma import Prisma
from passlib.context import CryptContext
from jose import JWTError, jwt
from pydantic import BaseModel, EmailStr
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT settings
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 1 week

class AppAuthService:
    def __init__(self):
        self.db = Prisma()
        
    async def create_user(self, user_data: BaseModel) -> Dict[str, Any]:
        """Create a new user with app credentials"""
        try:
            # Connect to database
            await self.db.connect()
            
            # Check if user already exists
            existing_user = await self.db.users.find_first(
                where={
                    "email": user_data.email
                }
            )
            
            if existing_user:
                return {"error": "Email already registered"}
                
            # Hash password
            hashed_password = pwd_context.hash(user_data.password)
            
            # Get max user ID to avoid conflicts
            max_id_result = await self.db.query_first(
                "SELECT MAX(user_id) as max_id FROM \"Users\""
            )
            
            next_id = 1
            if max_id_result and max_id_result.get('max_id') is not None:
                next_id = max_id_result['max_id'] + 1
            
            logger.info(f"Creating new user with ID {next_id}")
            
            # Create user with explicit ID
            user = await self.db.users.create(
                data={
                    "id": next_id,
                    "email": user_data.email,
                    "password": hashed_password,
                    "first_name": user_data.first_name,
                    "last_name": user_data.last_name,
                    "location": user_data.address,
                    "birth_date": user_data.birth_date,
                    "gender": user_data.gender,
                    "profile_pic": "",  # Default empty string
                    "bio": "",  # Default empty string
                    "preferred_sports": [],  # Default empty list
                    "location_lat": 0.0,  # Default value
                    "location_long": 0.0,  # Default value
                    "is_verified": False  # Default value
                }
            )
            
            # Generate access token
            access_token = self._create_access_token(user.id)
            
            return {"token": access_token}
            
        except Exception as e:
            logger.error(f"Error creating user: {str(e)}")
            return {"error": str(e)}
        finally:
            await self.db.disconnect()
            
    async def authenticate_user(self, user_data: BaseModel) -> Dict[str, Any]:
        """Authenticate user with email and password"""
        try:
            # Connect to database
            await self.db.connect()
            
            # Find user by email
            user = await self.db.users.find_first(
                where={
                    "email": user_data.email
                }
            )
            
            if not user:
                return {"error": "Invalid email or password"}
                
            # Verify password
            if not pwd_context.verify(user_data.password, user.password):
                return {"error": "Invalid email or password"}
                
            # Generate access token
            access_token = self._create_access_token(user.id)
            
            return {"token": access_token}
            
        except Exception as e:
            logger.error(f"Error authenticating user: {str(e)}")
            return {"error": str(e)}
        finally:
            await self.db.disconnect()
            
    def _create_access_token(self, user_id: int) -> str:
        """Create JWT access token"""
        expires_delta = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        expire = datetime.utcnow() + expires_delta
        
        to_encode = {
            "sub": str(user_id),
            "exp": expire
        }
        return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)