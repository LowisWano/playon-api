from fastapi import WebSocket, APIRouter
from config.ws_connection import *
from typing import List
from services.chats.user_chat_services import *
from services.socket_services import room_manager, RoomManagerPayload

user_chat: List[ChatConnection] = []
router = APIRouter(
    prefix="/user-chat",
    tags=["user-chat"]    
)

# frontend will generate the room id if it doesnt exist ( no conversation yet )
# and send to backend to insert the room id 
# @router.websocket("/ws/{room}/{client_id}")
@router.websocket("/ws/user-chat/{room}/{client_id}/{client_name}/{profile}")
async def user_chat_websocket(websocket: WebSocket, room: int, client_id: int, client_name: str, profile: str):
    payload = RoomManagerPayload(
        room=room,
        client_id=client_id,
        client_name=client_name,
        profile=profile,
        room_type="user-chat",
        chat_list=user_chat
    )
    await room_manager(payload, websocket)

@router.get("/get-all-chatmate/{user_id}")
async def get_all_chatmate_endpoint(user_id: int):
    return await get_all_chatmate(user_id)

@router.get("/get-chatmate/{user_chat_id}")
async def get_chatmate_endpoint(user_chat_id: int):
    return await get_chatmate(user_chat_id)

@router.post("/create-conversation")
async def create_conversation_endpoint(payload: CreateConversationDTO):
    await create_conversation(payload)
    
@router.delete("/delete-message/{message_id}")
async def soft_delete_message_endpoint(message_id: int):
    return await soft_delete_message(message_id)

