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
            message_data = json.loads(data)
            msg: str = message_data.get("msg")
            sentTo: str | int = message_data.get("to")

            message_payload = SendMessageDTO(
                room_id=p["room"],
                sender_id=p["client_id"],
                sender_name=p["client_name"],
                content=msg
            )

            if p["room_type"] == "group-chat":
                temp_bd = await send_message_group_chat(message_payload, str(sentTo))
            else:
                print("HERE IN USER CHAT")
                temp_bd = await send_message_user_chat(message_payload, int(sentTo))


            temp_bd_dict = temp_bd.model_dump()  # Convert Pydantic model to dictionary

            print("TEMP BD", temp_bd_dict)
            print("READ MESSAGES", temp_bd_dict["content"])

            broadcast_data = {
                "id": temp_bd_dict["id"],
                "user_chat_id": temp_bd_dict["user_chat_id"],
                "group_chat_id": temp_bd_dict["group_chat_id"],
                "sender_id": temp_bd_dict["sender_id"],
                "content": temp_bd_dict["content"],
                "is_deleted": temp_bd_dict["is_deleted"],
                "sent_at": temp_bd_dict["sent_at"].isoformat(),
                "read_messages": [
                    {
                        "id": rm["id"],
                        "sent_to_id": rm["sent_to_id"],
                        "message_id": rm["message_id"],
                        "is_read": rm["is_read"],
                        "read_at": rm["read_at"].isoformat() if rm["read_at"] else None
                    }
                    for rm in temp_bd_dict["read_messages"]
                ] if temp_bd_dict["read_messages"] else []
            }
            # print("Broadcast", broadcast_data)
                
            # broadcast to all users that are in the room
            for user in p["chat_list"]:
                if user["room"] == p["room"]:
                    await user["websocket"].send_text(json.dumps(broadcast_data))

    except WebSocketDisconnect:
        p["chat_list"][:] = [user for user in p["chat_list"] if not (user["room"] == p["room"] and user["client_id"] == p["client_id"])]
