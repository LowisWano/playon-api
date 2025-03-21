from db.prisma import prisma
from utils.http_error import *
from services.chats.dto.chat_dto import SendMessageDTO, CreateConversationDTO
from services.notification_services import notify_user, CreateNotifDTO


async def send_message(payload: SendMessageDTO):
    try:
        msg_id = await notify_user(
            CreateNotifDTO(
                user_id=payload.sender_id,
                message=f"New message from {payload.sender_name}",
                notif_type="User Chat",
                redirect_link=f"/user-chat/{payload.room_id}"
            )
        )
        await prisma.readmessage.create(
            data={
                "sent_to_id": payload.room_id,
                "message_id": msg_id.id
            }
        )
        
        await prisma.message.create(
            data={
                'content': payload.content,
                'sender_id': payload.sender_id,
                'user_chat_id': payload.room_id
            }
        )
    except Exception as e:
        raise BadRequestErrorResponse(error_message=str(e)).as_exception()


async def get_all_chatmate(user_id: int):
    chatmates = await prisma.userchat.find_many(
        where={
            "OR": [
                {"member1_id": user_id},
                {"member2_id": user_id}
            ]
        },
        # Prisma in Python Doesnt have selecting a specific column and
        # selecting a relationship if sender/receiver is equal to userid
        # let frontend do this
        include={
            "messages": {
                "take": 1,
                "orderBy": {
                    "sent_at": "desc"
                },
                "include": {
                    "read_messages": True
                }
            },
            "member1": True,
            "member2": True
        }
    )
    return chatmates


    
async def get_chatmate(user_chat_id):
    return await prisma.userchat.find_many(
        where={
            "id": user_chat_id
        },
        include={
            "messages": {
                "include": {
                    "read_messages": True
                }
            },
            "member1": True,
            "member2": True
        }
    )
    
# frontend will validate if both dont have conversation yet
# will trigger once the user sends the message
async def create_conversation(payload: CreateConversationDTO):
    try:
        await prisma.userchat.create(
            data={
                'member1_id': payload.sender_id,
                'member2_id': payload.receiver_id,
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