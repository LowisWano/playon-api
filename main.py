from utils.http_error import InternalServerErrorResponse
from fastapi import FastAPI
from db.prisma import prisma
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from routers.chats.group_chat_router import router as group_chat_router
from routers.chats.user_chat_router import router as user_chat_router
from test_routers import router
from routers.google_auth_router import router as google_auth_router
from routers.app_auth_router import router as app_auth_router
from routers.chats.read_message_router import router as read_message_router
from routers.sports_router import router as sports_router
from routers.recommender_router import router as recommender_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8081"],
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
    try:
        users = await prisma.users.find_many(
            include={
                "created_groups": True,
            }
        )
        return users
    except Exception as e:
        raise InternalServerErrorResponse(error_message=str(e)).as_exception()

def configure_routers(app=app):
    app.get("/health-check")(lambda: {"Hello": "World"})
    app.include_router(recommender_router)
    app.include_router(group_chat_router)
    app.include_router(user_chat_router)
    app.include_router(read_message_router)
    app.include_router(sports_router)
    app.include_router(google_auth_router)
    app.include_router(app_auth_router)
    app.include_router(router)      

configure_routers()