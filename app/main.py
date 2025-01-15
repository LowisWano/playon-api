from typing import Union
from utils.HttpError import *
from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


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
