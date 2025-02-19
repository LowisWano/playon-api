from db.prisma import prisma
from utils.http_error import *
from pydantic import BaseModel

# id in GroupChat will be use for room id in websockets
# some returns an any, give it a type

class CreateGroupChat(BaseModel):
    created_by: int
    title: str

class CreateGroupUsers(BaseModel):
    user_id: int
    class Config:
        orm_mode = True
    

async def get_group_chat_by_id(group_id: int):
    print("HEREE")
    return await prisma.groupchat.find_unique (
        where={"id": group_id},
        include={
            "messages": True,
            "group_users": {
                "include": {
                    "user": True
                }
            }
        }
    )


async def create_group_chat(payloadGC: CreateGroupChat, payloadGU: CreateGroupUsers):
    try:
        gc = await prisma.groupchat.create(
            data={
                'created_by': payloadGC.created_by,
                'title': payloadGC.title
            },
        )
        # Add the first user (the creator) to the group
        gu = await prisma.groupuser.create(
            data={
                'group_chat_id': gc.id,
                'user_id': payloadGU.user_id
            }
        )
    except Exception as e:
        raise BadRequestErrorResponse(error_message=str(e)).as_exception()

    # Fetch the full details of the newly created group chat
    createdGC = await get_group_chat_by_id(gc.id)

    if createdGC is None:
        raise BadRequestErrorResponse(error_message=str(e)).as_exception()
    
    return {
        "data": createdGC,
        "message": "Group Chat Created Successfully!"
    }

    