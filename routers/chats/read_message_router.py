from fastapi import APIRouter
from pydantic import BaseModel
from db.prisma import prisma
from services.chats.group_chat_services import get_all_user_group_chats
from services.chats.user_chat_services import get_all_chatmate


router = APIRouter(
    prefix="/read-message",
    tags=["read-message"]    
)

# messages screen in FE
@router.get("/get-all-messages-of-user/{user_id}")
async def get_all_messages_of_user_endpoint(user_id: int):
    gc = await get_all_user_group_chats(user_id)
    ch = await get_all_chatmate(user_id)
    return {
        "group_chats": gc,
        "chatmates": ch
    }

@router.delete("/read/{read_message_id}")
async def read_message_endpoint(read_message_id: int):
    return await prisma.readmessage.update(
        where={
            "id": read_message_id
        },
        data={
            "is_read": True
        }
    )
