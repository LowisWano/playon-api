from db.prisma import prisma
from routers.notification_router import CreateNotifDTO
from utils.http_error import BadRequestErrorResponse

async def get_all_notifications(user_id: int):
    return await prisma.notifications.find_many(
        where={
            "user_id": user_id
        },
        order={
            "id": "desc"
        }
    )

async def notify_user(p: CreateNotifDTO, msg_id: int = None):
    try:
        if p.notif_type == "Group Chat":
            gcMembers = await prisma.groupuser.find_many(
                where={
                    "group_chat_id": p.notif_to_id,
                    "isGcOnMute": False
                }
            )
            for members in gcMembers:
                await prisma.notifications.create(
                    data={
                        "notif_from_id": p.notif_from_id,
                        "notif_to_id": members.user_id,
                        "message": p.message,
                        "type": p.notif_type,
                        "redirect_link": p.redirect_link
                    }
                )
                await prisma.readmessage.create(
                    data={
                        "sent_to_id": members.user_id,
                        "message_id": msg_id
                    }
                )
            return
            
        
        # when type is GrpChat, room_id is used for notif_to
        await prisma.notifications.create(
            data={
                "notif_from_id": p.notif_from_id,
                "notif_to_id": p.notif_to_id,
                "message": p.message,
                "type": p.notif_type,
                "redirect_link": p.redirect_link
            }
        )
    except Exception as e:
        raise BadRequestErrorResponse(error_message=str(e)).as_exception()

async def read_notification(notification_id: int):
    try:
        return await prisma.notifications.delete(
            where={
                "id": notification_id
            }
        )
    except Exception as e:
        raise BadRequestErrorResponse(error_message=str(e)).as_exception()