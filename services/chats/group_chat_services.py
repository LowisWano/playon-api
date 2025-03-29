from db.prisma import prisma
from utils.http_error import BadRequestErrorResponse
from services.chats.dto.chat_dto import *
from services.notification_services import notify_user, CreateNotifDTO
from typing import TypedDict

# id in GroupChat will be use for room id in websockets

async def send_message(payload: SendMessageDTO, gc_name: str):
    try:
        msg = await prisma.message.create(
            data={
                'content': payload.content,
                'sender_id': payload.sender_id,
                'group_chat_id': payload.room_id
            },
        )    
        return msg
    except Exception as e:
        raise BadRequestErrorResponse(error_message=str(e)).as_exception()
    finally:
        await notify_user(
            CreateNotifDTO(
                user_id=payload.room_id,
                message=f"New message from {gc_name}",
                notif_type="Group Chat",
                redirect_link=f"/group-chat/{payload.room_id}"
            ),
            msg_id=msg.id
        )

async def get_all_user_group_chats(user_id: int):    
    group_chats = await prisma.groupuser.find_many(
        where={
            "user_id": user_id
        },
        include={
            "group_chat": {
                "include": {
                    "messages": {
                        "take": 1,
                        "orderBy": {
                            "sent_at": "desc"
                        },
                        "include": {
                            "read_messages": True
                        }
                    }
                },
            },
        }
    )
    return group_chats

############################################
######## FUCK PYTHON PRISMA ################
############################################

async def get_group_chat_by_id(group_id: int):
    query = """
        SELECT 
            gc.group_chat_id,
            gc.created_by,
            gc.gc_name,
            gc.group_pic,
            gc.created_at,
            (
                SELECT json_agg(json_build_object(
                    'id', gu.group_chat_user_id,
                    'user_id', gu.user_id,
                    'is_pending', gu.is_pending,
                    'is_on_mute', gu.is_on_mute,
                    'joined_at', gu.joined_at,
                    'user', (
                        SELECT json_build_object(
                            'id', u.user_id,
                            'first_name', u.first_name,
                            'last_name', u.last_name,
                            'profile_pic', u.profile_pic
                        )
                        FROM "Users" u
                        WHERE u.user_id = gu.user_id
                    )
                ))
                FROM "GroupUser" gu
                WHERE gu.group_chat_id = gc.group_chat_id
            ) as group_users,
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
                ) ORDER BY m.sent_at ASC)
                FROM "Message" m
                WHERE m.group_chat_id = gc.group_chat_id
            ) as messages
        FROM "GroupChat" gc
        WHERE gc.group_chat_id = $1
    """
    result = await prisma.query_raw(query, group_id)
    return result[0] if result else None


async def search_users_not_in_group_chat(group_id: int, username: str, user_id: int):
    query = """
        SELECT 
            u.user_id,
            u.first_name,
            u.last_name,
            u.profile_pic
        FROM "Users" u
        WHERE u.user_id NOT IN (
            SELECT gu.user_id
            FROM "GroupUser" gu
            WHERE gu.group_chat_id = $1
        )
        AND u.user_id <> $3
        AND LOWER(CONCAT(u.first_name, ' ', u.last_name)) LIKE LOWER($2)
        LIMIT 15
    """
    search_pattern = f"%{username}%"
    return await prisma.query_raw(query, group_id, search_pattern, user_id)


async def recent_contacts_not_in_group_chat(group_id: int, user_id: int):
    query = """
        SELECT
            u.user_id,
            u.first_name,
            u.last_name,
            u.profile_pic
        FROM "Users" u
        WHERE u.user_id NOT IN (
            SELECT gu.user_id
            FROM "GroupUser" gu
            WHERE gu.group_chat_id = $1
        )
        AND u.user_id != $2
        AND (
            u.user_id IN (
                SELECT uc.member1_id
                FROM "UserChat" uc
                WHERE uc.member2_id = $2
            )
            OR u.user_id IN (
                SELECT uc.member2_id
                FROM "UserChat" uc
                WHERE uc.member1_id = $2
            )
        )
    """
    return await prisma.query_raw(query, group_id, user_id)

async def add_to_group_chat(p: AddToGroupChatDTO):
    try:
        autoAccept = True if p.isAutoApprove else (True if p.isAdmin else False)
        await prisma.groupuser.create(
            data={
                'group_chat_id': p.group_id,
                'user_id': p.user_id,
                'isPending': autoAccept
            }
        )
    except Exception as e:
        raise BadRequestErrorResponse(error_message=str(e)).as_exception()
    
    

async def create_group_chat(payloadGC: CreateGroupChatDTO):
    try:
        gc = await prisma.groupchat.create(
            data={
                'created_by': payloadGC.created_by,
                'gc_name': payloadGC.gc_name,
                'group_pic': payloadGC.image
            },
        )
        # add the first user (the creator) to the group
        await prisma.groupuser.create(
            data={
                'group_chat_id': gc.id,
                'user_id': payloadGC.created_by
            }
        )

        createdGC = await get_group_chat_by_id(gc.id)

        if createdGC is None:
            raise BadRequestErrorResponse(error_message=str(e)).as_exception()
        
        return {
            "data": createdGC,
            "message": "Group Chat Created Successfully!"
        }
    except Exception as e:
        raise BadRequestErrorResponse(error_message=str(e)).as_exception()
    
    
async def mute_groupchat_for_user(group_user_id: int):
    try:
        await prisma.groupuser.update(
            where={
                'id': group_user_id
            },
            data={
                'isGcOnMute': True
            }
        )
    except Exception as e:
        raise BadRequestErrorResponse(error_message=str(e)).as_exception()
    
    
async def accept_user_to_gc(group_user: int):
    try:
        await prisma.groupuser.update(
            where={
                'id': group_user
            },
            data={
                'isPending': False
            }
        )
    except Exception as e:
        raise BadRequestErrorResponse(error_message=str(e)).as_exception()


async def transfer_creator_role(gc_id: int, new_user_creator_id: int):
    try:
        await prisma.groupchat.update(
            where={
                "id": gc_id
            },
            data={
                "created_by": new_user_creator_id
            }
        )
        return {
            "message": "Admin role transfered succesfully!"
        }
    except Exception as e:
        raise BadRequestErrorResponse(error_message=str(e)).as_exception()

  
# frontend will filter if admin or not to prevent leaving
async def leave_group_chat(grp_member_id: int):
    try:
        await prisma.groupuser.delete(
            where={
                "id": grp_member_id
            }
        )
        return {
            "message": "You left the group chat successfully!"
        }
    except Exception as e:
        raise BadRequestErrorResponse(error_message=str(e)).as_exception()
    
  
# only admin can kick | filtered in FE
async def kick_group_chat_member(grp_user_id: int):
    try:
        await leave_group_chat(grp_user_id)
        return {
            "message": "User kicked successfully!"
        }
    except Exception as e:
        raise BadRequestErrorResponse(error_message=str(e)).as_exception()
    
    
# only admin can delete when gc is now empty | will filtered in FE
async def delete_group_chat(gc_id: int):
    try:
        await prisma.message.delete_many(
            where={
                "group_chat_id": gc_id
            }
        )
        
        await prisma.groupchat.delete(
            where={
                "id": gc_id
            }
        )
        
        return {
            "message" : "Group chat deleted successfully!"
        }
    except Exception as e:
        raise BadRequestErrorResponse(error_message=str(e)).as_exception()
            