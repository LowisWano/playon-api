from db.prisma import prisma
from utils.http_error import *
from services.chats.dto.chat_dto import SendMessageDTO, CreateConversationDTO
from services.notification_services import notify_user, CreateNotifDTO

async def send_message(payload: SendMessageDTO, sent_to_id: int):
    try:
        msg = await prisma.message.create(
            data={
                'content': payload.content,
                'sender_id': payload.sender_id,
                'user_chat_id': payload.room_id
            },
        )
        await prisma.readmessage.create(
            data={
                "sent_to_id": sent_to_id,
                "message_id": msg.id
            }
        )
        msg = await prisma.message.find_unique(
            where={
                "id": msg.id
            },
            include={
                "read_messages": True
            }
        )
        return msg
    except Exception as e:
        raise BadRequestErrorResponse(error_message=str(e)).as_exception()
    # finally:
        # await notify_user(
        #     CreateNotifDTO(
        #         user_id=payload.sender_id,
        #         message=f"New message from {payload.sender_name}",
        #         notif_type="User Chat",
        #         redirect_link=f"/user-chat/{payload.room_id}"
        #     )
        # )


async def get_all_chatmate(user_id: int):
    query = """
        SELECT 
            uc.chat_id,
            uc.member1_id,
            uc.member2_id,
            uc.room_id,
            uc.created_at,
            -- Member 1 details
            (
                SELECT json_build_object(
                    'user_id', m1.user_id,
                    'first_name', m1.first_name,
                    'last_name', m1.last_name,
                    'profile_pic', m1.profile_pic
                )
                FROM "Users" m1
                WHERE m1.user_id = uc.member1_id
            ) as member1,
            -- Member 2 details
            (
                SELECT json_build_object(
                    'user_id', m2.user_id,
                    'first_name', m2.first_name,
                    'last_name', m2.last_name,
                    'profile_pic', m2.profile_pic
                )
                FROM "Users" m2
                WHERE m2.user_id = uc.member2_id
            ) as member2,
            -- Latest message
            (
                SELECT json_agg(json_build_object(
                    'id', m.message_id,
                    'user_chat_id', m.user_chat_id,
                    'sender_id', m.sender_id,
                    'content', m.content,
                    'group_chat_id', m.group_chat_id,
                    'is_deleted', m.is_deleted,
                    'sent_at', m.sent_at,
                    'read_messages', (
                        SELECT json_agg(json_build_object(
                            'id', rm.read_message_id,
                            'sent_to_id', rm.sent_to_id,
                            'is_read', rm.is_read,
                            'read_at', rm.read_at
                        ))
                        FROM "ReadMessage" rm
                        WHERE rm.message_id = m.message_id
                    )
                ))
                FROM "Message" m
                WHERE m.user_chat_id = uc.chat_id
                GROUP BY m.message_id
                ORDER BY m.message_id DESC
                LIMIT 1
            ) as messages
        FROM "UserChat" uc
        WHERE uc.member1_id = $1 OR uc.member2_id = $1
    """
    
    # Execute the query with the user_id as a parameter
    result = await prisma.query_raw(query, user_id)
    
    # Return the result
    return result if result else []



    
async def get_chatmate(user_chat_id: int):
    return await prisma.userchat.find_unique(
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
                'room_id': str(payload.room_id)
            }
        )
        await send_message(
            SendMessageDTO(
                room_id=payload.room_id,
                sender_id=payload.sender_id,
                sender_name=payload.sender_name,
                content=payload.content
            ),
            sent_to_id=payload.receiver_id
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