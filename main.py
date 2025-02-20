from utils.http_error import *
from fastapi import FastAPI
from db.prisma import prisma
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from routers.group_chat_router import router as group_chat_router
from test_routers import router
from routers.google_auth import router as google_auth_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup():
    await prisma.connect()
    
@app.on_event("shutdown")
async def shutdown():
    await prisma.disconnect()

@app.get("/check-db")
async def check_db():
    users = await prisma.userchat.find_many(
        include={
            "messages": True,
            "sender": True,
            "receiver": True
        }
    )
    return users


def configure_routers(app=app):
    app.get("/health-check")(lambda: {"Hello": "World"})
    app.include_router(group_chat_router)
    app.include_router(google_auth_router)
    app.include_router(router)      

configure_routers()