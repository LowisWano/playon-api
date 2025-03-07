from db.prisma import prisma
from utils.http_error import *
from services.chats.dto.chat_dto import SendMessageDTO, CreateConversationDTO
from services.notification_services import notify_user, CreateNotifDTO


async def send_message(payload: SendMessageDTO):
    await notify_user(
        CreateNotifDTO(
            user_id=payload.sender_id,
            message=f"New message from {payload.sender_name}",
            notif_type="User Chat",
            redirect_link=f"/user-chat/{payload.room_id}"
        )
    )
    await prisma.message.create(
        data={
            'content': payload.content,
            'sender_id': payload.sender_id,
            'user_chat_id': payload.room_id
        }
    )
    

async def get_all_chatmate(user_id: int):
    return await prisma.userchat.find_many(
        where={
            "OR": [
                {"sender_id": user_id},
                {"receiver_id": user_id}
            ]
        },
        # Prisma in Python Doesnt have selecting a specific column and
        # selecting a relationship if sender/receiver is equal to userid
        # let frontend do this
        include={
            "sender": True, 
            "receiver": True
        }
    )

    
async def get_chatmate(user_chat_id):
    return await prisma.message.find_many(
        where={
            "user_chat_id": user_chat_id
        }
    )
    
# frontend will validate if both dont have conversation yet
# will trigger once the user sends the message
async def create_conversation(payload: CreateConversationDTO):
    try:
        await prisma.userchat.create(
            data={
                'sender_id': payload.sender_id,
                'receiver_id': payload.receiver_id,
                'room_id': payload.room_id
            }
        )
        await prisma.userchat.create(
            data={
                'sender_id': payload.receiver_id,
                'receiver_id': payload.sender_id,
                'room_id': payload.room_id
            }
        )
    except Exception as e:
        raise BadRequestErrorResponse(error_message=str(e)).as_exception()
   
async def soft_delete_message(message_id: int):
    try:
        return await prisma.message.update(
            where={
                "id": message_id
            },
            data={
                "is_deleted": True
            }
        )
    except Exception as e:
        raise BadRequestErrorResponse(error_message=str(e)).as_exception()