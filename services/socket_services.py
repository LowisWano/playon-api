from fastapi import WebSocket
from fastapi.websockets import WebSocketDisconnect
from services.chats.group_chat_services import send_message as send_message_group_chat
from services.chats.user_chat_services import send_message as send_message_user_chat
from typing import List
import json
from config.ws_connection import *
from typing import TypedDict
from services.chats.dto.chat_dto import SendMessageDTO
from pydantic import ConfigDict

manager = ConnectionManager()

class ChatContent(TypedDict):
    id: int
    profile: str
    username: str
    message: str
    user: ChatConnection
    
class RoomManagerPayload(TypedDict):
    room: int
    client_id: int
    client_name: str
    profile: str
    room_type: str
    chat_list: List[ChatConnection]

async def room_manager(p: RoomManagerPayload, websocket: WebSocket):
    await manager.connect(websocket)

    # check if the room exists, initialize it if not
    if not any(connection["room"] == p["room"] for connection in p["chat_list"]):
        p["chat_list"].append({
            "room": p["room"],
            "client_id": p["client_id"],
            "websocket": websocket
        })
    else:
        # if the room already exists just append the new user
        p["chat_list"].append({
            "room": p["room"],
            "client_id": p["client_id"],
            "websocket": websocket
        })

    print(f"Users in room {p['room']}: {[user['client_id'] for user in p['chat_list'] if user['room'] == p['room']]}")

    try:
        while True:
            data = await websocket.receive_text()
            print(data)

            content: ChatContent = {
                "id": p["client_id"],
                "profile": p["profile"],
                "username": p["client_name"],
                "message": data,
                "user": websocket
            }
            
            broadcast_data = {
                "id": content["id"],
                "profile": content["profile"],
                "username": content["username"],
                "message": content["message"]
            }

            # broadcast to all users that are in the room
            for user in p["chat_list"]:
                if user["room"] == p["room"]:
                    await user["websocket"].send_text(json.dumps(broadcast_data))

            message_payload = SendMessageDTO(
                chat_id=p["room"],
                sender_id=content["id"],
                content=content["message"]
            )

            if p["room_type"] == "group-chat":
                await send_message_group_chat(message_payload)
            elif p["room_type"] == "user-chat":
                await send_message_user_chat(message_payload)

    except WebSocketDisconnect:
        p["chat_list"][:] = [user for user in p["chat_list"] if not (user["room"] == p["room"] and user["client_id"] == p["client_id"])]
