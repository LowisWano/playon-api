from fastapi import WebSocket, APIRouter
from fastapi.websockets import WebSocketDisconnect
from config.ws_connection import ConnectionManager


# stores here rooms with its users
# so that we can broadcast messages to the correct room
# not yet implemented
user_chat = []

manager = ConnectionManager()

router = APIRouter()

@router.websocket("/ws/user-chat/{room}/{client_id}")
async def user_chat_websocket(websocket: WebSocket, room: str, client_id: int):
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
    