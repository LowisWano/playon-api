import requests
from jose import jwt
import os
from dotenv import load_dotenv

load_dotenv()

class GoogleAuthService:
    def __init__(self):
        self.client_id = os.getenv("GOOGLE_CLIENT_ID")
        self.client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
        self.redirect_uri = os.getenv("GOOGLE_REDIRECT_URI")

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

    def decode_token(self, token: str):
        return jwt.decode(token, self.client_secret, algorithms=["HS256"])