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
@router.websocket("/ws/{room}/{gc_name}/{client_id}/{client_name}/{profile}")
async def group_chat_websocket(websocket: WebSocket, room: int, gc_name: str, client_id: int, client_name: str, profile: str):
    payload = RoomManagerPayload(
        room=room,
        client_id=client_id,
        client_name=client_name,
        profile=profile,
        room_type="group-chat",
        chat_list=group_chat
    )
    await room_manager(payload, websocket, gc_name)

@router.get("/users-gc/{user_id}")
async def get_all_user_group_chats_endpoint(user_id: int):
    return await get_all_user_group_chats(user_id)

@router.get("/visit/{group_id}")
async def get_group_chat_endpoint(group_id: int):
    return await get_group_chat_by_id(group_id);

@router.post("/create-gc")
async def create_group_chat_endpoint(payloadGC: CreateGroupChatDTO):
    return await create_group_chat(payloadGC)

@router.post("/add-to-groupchat")
async def add_to_group_chat_endpoint(payload: AddToGroupChatDTO):
    return await add_to_group_chat(payload)

@router.patch("/accept-user/{group_user}")
async def accept_user_to_gc_endpoint(group_user: int):
    return await accept_user_to_gc(group_user);

@router.patch("/mute-gc/{group_user_id}")
async def mute_groupchat_for_user_endpoint(group_user_id: int):
    return await mute_groupchat_for_user(group_user_id);

@router.patch("/group-chat/{gc_id}/{new_user_creator_id}")
async def transfer_creator_role_endpoint(gc_id: int, new_user_creator_id: int):
    return await transfer_creator_role(gc_id, new_user_creator_id)

@router.delete("/kick-member/{group_member_id}")
async def kick_group_chat_member(group_member_id: int):
    return await kick_group_chat_member(group_member_id)

@router.delete("/delete-gc/{gc_id}")
async def delete_group_chat(gc_id: int):
    return await delete_group_chat(gc_id)

@router.delete("/leave-gc/{group_member_id}")
async def leave_group_chat_endpoint(group_member_id: int):
    return await leave_group_chat(group_member_id)
