from db.prisma import prisma
from utils.http_error import *
from services.chats.dto.chat_dto import *

# id in GroupChat will be use for room id in websockets

async def send_message(payload: SendMessageDTO):
    await prisma.message.create(
        data={
            'content': payload.content,
            'sender_id': payload.sender_id,
            'group_chat_id': payload.chat_id
        }
    )    

async def get_all_user_group_chats(user_id: int):
    return await prisma.groupuser.find_many(
        where={
            "user_id": user_id
        },
        include={
            "group_chat": True
        }
    )


async def get_group_chat_by_id(group_id: int):
    return await prisma.groupchat.find_unique (
        where = {
            "id": group_id
        },
        include = {
            "messages": True,
            "group_users": {
                "include": {
                    "user": True
                }
            }
        }
    )


async def create_group_chat(payloadGC: CreateGroupChatDTO):
    try:
        gc = await prisma.groupchat.create(
            data={
                'created_by': payloadGC.created_by,
                'title': payloadGC.title
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
            