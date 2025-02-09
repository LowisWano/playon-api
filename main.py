from typing import Union
from utils.HttpError import *
from fastapi import FastAPI
from db.prisma import prisma

app = FastAPI()

@app.on_event("startup")
async def startup():
    await prisma.connect()
    
@app.on_event("shutdown")
async def shutdown():
    await prisma.disconnect()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/check-db")
async def check_db():
    users = await prisma.user.find_many()
    return users

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


@app.get("/error-check")
def error_check():
    x = 0
    try:
        return {"result": 100 / x}
    except Exception as e:
        print(e, "HEY")
        return BadRequestErrorResponse(error_message=str(e)).as_exception()
