from fastapi import WebSocket, APIRouter
from fastapi.websockets import WebSocketDisconnect
from config.ws_connection import ConnectionManager
from services.chats.group_chat_services import *
from typing import List, TypedDict 
import json

# stores here the rooms and users that are in the group chat
# so that we can broadcast messages to the correct room
class GroupChatConnection(TypedDict):
    room: int
    client_id: int
    websocket: WebSocket
    
group_chat: List[GroupChatConnection] = []

manager = ConnectionManager()
router = APIRouter(prefix="/group-chat")

@router.websocket("/ws/{room}/{client_id}/{client_name}/{profile}")
async def group_chat_websocket(websocket: WebSocket, room: int, client_id: int):
    
    await manager.connect(websocket)

    # Check if the room exists, if not, initialize it
    if not any(connection["room"] == room for connection in group_chat):
        group_chat.append({
            "room": room,
            "client_id": client_id,
            "websocket": websocket
        })
    else:
        # If the room already exists, just append the new user
        group_chat.append({
            "room": room,
            "client_id": client_id,
            "websocket": websocket
        })
    
    # Debug
    print(f"Users in room {room}: {[user['client_id'] for user in group_chat if user['room'] == room]}")
    
    try:
        while True:
            data = await websocket.receive_text()
            for user in group_chat:
                if user["room"] == room:
                    content = {
                        "id": client_id,
                        "profile": "HEY",
                        "username": "test",
                        "message": data
                    }
                    await insert_to_messages(room, client_id, data)
                    await user["websocket"].send_text(json.dumps(content))
    except WebSocketDisconnect:
        # Remove the user from the room when they disconnect
        group_chat[:] = [user for user in group_chat if not (user["room"] == room and user["client_id"] == client_id)]

@router.get("/users-gc/{user_id}")
async def get_all_user_group_chats_endpoint(user_id: int):
    data = await get_all_user_group_chats(user_id)
    return data

@router.get("/visit/{group_id}")
async def get_group_chat_endpoint(group_id: int):
    data = await get_group_chat_by_id(group_id);
    return data

@router.post("/create-gc")
async def create_group_chat_endpoint(payloadGC: CreateGroupChatDTO, payloadGU: CreateGroupUsersDTO):
    data = await create_group_chat(payloadGC, payloadGU)
    return data

@router.patch("/group-chat/{gc_id}/{new_user_creator_id}")
async def transfer_creator_role_endpoint(gc_id: int, new_user_creator_id: int):
    data = await transfer_creator_role(gc_id, new_user_creator_id)
    return data

@router.delete("/kick-member/{group_member_id}")
async def kick_group_chat_member(group_member_id: int):
    data = kick_group_chat_member(group_member_id)
    return data

@router.delete("/delete-gc/{gc_id}")
async def delete_group_chat(gc_id: int):
    data = delete_group_chat(gc_id)
    return data

@router.delete("/leave-gc/{group_member_id}")
async def leave_group_chat_endpoint(group_member_id: int):
    data = await leave_group_chat(group_member_id)
    return data

