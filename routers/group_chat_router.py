from fastapi import WebSocket, APIRouter
from fastapi.websockets import WebSocketDisconnect
from config.ws_connection import ConnectionManager
from services.chats.group_chat_services import *
from pydantic import BaseModel

# stores here the rooms and users that are in the group chat
# so that we can broadcast messages to the correct room
# not yet implemented
group_chat = []

class CreateGroupChat(BaseModel):
    created_by: int
    title: str

class CreateGroupUsers(BaseModel):
    user_id: int
    class Config:
        orm_mode = True

manager = ConnectionManager()
router = APIRouter()

@router.websocket("/ws/group-chat/{room}/{client_id}")
async def group_chat_websocket(websocket: WebSocket, room: str, client_id: int):
    print(f"Connection from {websocket.headers.get('origin')}")
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            print(data)
            await manager.broadcast(f"Client #{client_id} says: {data}, room: {room}") # test
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast(f"Client #{client_id} left the chat")

@router.get("/group-chat/{group_id}")
async def get_group_chat_endpoint(group_id: int):
    if group_id is None:
        return BadRequestErrorResponse(error_message="Group ID is required").as_response()
    data = await get_group_chat_by_id(group_id);
    if data is None:
        return { "message": "Group Chat Doesn't Exists." }
    return data

@router.post("/create-gc")
async def create_group_chat_endpoint(payloadGC: CreateGroupChat, payloadGU: CreateGroupUsers):
    data = await create_group_chat(payloadGC=payloadGC, payloadGU=payloadGU)
    return data