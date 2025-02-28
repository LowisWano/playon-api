from fastapi import WebSocket, APIRouter
from config.ws_connection import *
from services.chats.group_chat_services import *
from typing import List
from services.socket_services import room_manager, RoomManagerPayload

# id in GroupChat will be use for room id in websockets

group_chat: List[ChatConnection] = []
router = APIRouter(
    prefix="/group-chat",
    tags=["group-chat"]    
)

# @router.websocket("/ws/{room}/{client_id}")
@router.websocket("/ws/{room}/{client_id}/{client_name}/{profile}")
async def group_chat_websocket(websocket: WebSocket, room: int, client_id: int, client_name: str, profile: str):
    payload = RoomManagerPayload(
        room=room,
        client_id=client_id,
        client_name=client_name,
        profile=profile,
        room_type="group-chat",
        chat_list=group_chat
    )
    await room_manager(payload, websocket)

@router.get("/users-gc/{user_id}")
async def get_all_user_group_chats_endpoint(user_id: int):
    data = await get_all_user_group_chats(user_id)
    return data

@router.get("/visit/{group_id}")
async def get_group_chat_endpoint(group_id: int):
    data = await get_group_chat_by_id(group_id);
    return data

@router.post("/create-gc")
async def create_group_chat_endpoint(payloadGC: CreateGroupChatDTO):
    data = await create_group_chat(payloadGC)
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

