import requests
from jose import jwt
import os
from dotenv import load_dotenv
from prisma import Prisma
from datetime import datetime, timedelta
from fastapi import HTTPException
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class GoogleAuthService:
    def __init__(self):
        self.client_id = os.getenv("GOOGLE_CLIENT_ID")
        self.client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
        self.redirect_uri = os.getenv("GOOGLE_REDIRECT_URI")
        self.jwt_secret = os.getenv("JWT_SECRET_KEY")
        self.prisma = Prisma()

    def get_auth_url(self):
        return f"https://accounts.google.com/o/oauth2/auth?response_type=code&client_id={self.client_id}&redirect_uri={self.redirect_uri}&scope=openid%20profile%20email&access_type=offline"

    async def get_tokens(self, code: str):
        token_url = "https://accounts.google.com/o/oauth2/token"
        data = {
            "code": code,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "redirect_uri": self.redirect_uri,
            "grant_type": "authorization_code",
        }
        response = requests.post(token_url, data=data)
        return response.json()

    async def get_user_info(self, access_token: str):
        response = requests.get(
            "https://www.googleapis.com/oauth2/v1/userinfo",
            headers={"Authorization": f"Bearer {access_token}"}
        )
        return response.json()

    def create_jwt_token(self, user_id: int) -> str:
        expires_delta = timedelta(days=7)
        expire = datetime.utcnow() + expires_delta
        
        to_encode = {
            "exp": expire,
            "sub": str(user_id)
        }
        return jwt.encode(to_encode, self.jwt_secret, algorithm="HS256")

    def decode_token(self, token: str):
        return jwt.decode(token, self.jwt_secret, algorithms=["HS256"])

    async def authenticate_google_user(self, user_info: dict):
        """Handle both login and signup for Google users"""
        try:
            await self.prisma.connect()
            
            # Check if user exists with the given email
            existing_user = await self.prisma.users.find_first(
                where={
                    "email": user_info["email"]
                }
            )

            if existing_user:
                logger.info(f"Found existing user with email {user_info['email']}")
                if existing_user.provider_id == user_info["id"]:
                    # User exists and provider ID matches - return JWT
                    return {"token": self.create_jwt_token(existing_user.id)}
                else:
                    # Email exists but different provider - return error
                    return {"error": "Email already exists with different provider"}

            # Check if provider_id exists
            existing_provider = await self.prisma.users.find_first(
                where={
                    "provider_id": user_info["id"]
                }
            )

            if existing_provider:
                logger.info(f"Found existing user with provider ID {user_info['id']}")
                # Provider ID exists but email doesn't match - return error
                return {"error": "Provider ID already exists with different email"}

            # Create new user with minimal required fields
            try:
                # Get max user ID to avoid conflicts
                max_id_result = await self.prisma.query_first(
                    "SELECT MAX(user_id) as max_id FROM \"Users\""
                )
                
                next_id = 1
                if max_id_result and max_id_result.get('max_id') is not None:
                    next_id = max_id_result['max_id'] + 1
                
                logger.info(f"Creating new user with ID {next_id}")
                
                # Create the user with explicit ID
                new_user = await self.prisma.users.create(
                    data={
                        "id": next_id,
                        "email": user_info["email"],
                        "password": "",  # Empty password for Google auth
                        "provider_id": user_info["id"],
                        "first_name": user_info.get("given_name", ""),
                        "last_name": user_info.get("family_name", ""),
                        "profile_pic": user_info.get("picture", ""),
                        "birth_date": datetime.utcnow(),  # Placeholder date
                        "gender": "OTHER",  # Default gender
                        "bio": "",
                        "is_verified": True,  # Auto verify Google users
                        "preferred_sports": [],
                        "location": "",
                        "location_lat": 0.0,
                        "location_long": 0.0
                    }
                )
                logger.info(f"User created successfully with ID {new_user.id}")
                return {"token": self.create_jwt_token(new_user.id)}
                
            except Exception as create_error:
                logger.error(f"Error creating user: {str(create_error)}")
                
                # If user creation fails, try to find if user was actually created
                created_user = await self.prisma.users.find_first(
                    where={
                        "email": user_info["email"]
                    }
                )
                
                if created_user:
                    logger.info(f"User was created despite error, ID: {created_user.id}")
                    return {"token": self.create_jwt_token(created_user.id)}
                
                # Try creating with raw SQL as a fallback
                try:
                    logger.info("Attempting to create user with raw SQL")
                    result = await self.prisma.query_first(
                        """
                        INSERT INTO "Users" (
                            "user_id", "email", "password", "provider_id", "first_name", "last_name", 
                            "birth_date", "gender", "bio", "profile_pic", "is_verified", 
                            "preferred_sports", "location", "location_lat", "location_long"
                        ) 
                        VALUES (
                            (SELECT COALESCE(MAX("user_id"), 0) + 1 FROM "Users"), 
                            $1, $2, $3, $4, $5, $6, 'OTHER', $7, $8, $9, '{}', $10, $11, $12
                        )
                        RETURNING "user_id"
                        """,
                        user_info["email"],
                        "",
                        user_info["id"],
                        user_info.get("given_name", ""),
                        user_info.get("family_name", ""),
                        datetime.utcnow(),
                        "",
                        user_info.get("picture", ""),
                        True,
                        "",
                        0.0,
                        0.0
                    )
                    
                    if result and 'user_id' in result:
                        user_id = result['user_id']
                        logger.info(f"User created with raw SQL, ID: {user_id}")
                        return {"token": self.create_jwt_token(user_id)}
                    
                except Exception as sql_error:
                    logger.error(f"Raw SQL user creation failed: {str(sql_error)}")
                
                return {"error": f"Failed to create user: {str(create_error)}"}

        except Exception as e:
            logger.error(f"Error in authenticate_google_user: {str(e)}")
            return {"error": str(e)}
        finally:
            await self.prisma.disconnect()

    async def process_google_callback(self, code: str):
        """Process the Google OAuth callback and create/login the user"""
        try:
            # Exchange code for tokens
            tokens = await self.get_tokens(code)
            if "error" in tokens:
                return {"error": tokens.get("error_description", "Failed to get tokens")}
            
            # Get user info from Google
            user_info = await self.get_user_info(tokens["access_token"])
            if "error" in user_info:
                return {"error": "Failed to get user info from Google"}
            
            # Authenticate/create the user
            result = await self.authenticate_google_user(user_info)
            return result
            
        except Exception as e:
            logger.error(f"Error in process_google_callback: {str(e)}")
            return {"error": str(e)}