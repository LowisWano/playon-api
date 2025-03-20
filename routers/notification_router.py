from fastapi import APIRouter
from services.notification_services import *
from pydantic import BaseModel

router = APIRouter(
    prefix="/notification",
    tags=["notification"]    
)

class CreateNotifDTO(BaseModel):
    notif_to_id: int
    notif_from_id: int
    message: str
    notif_type: int
    redirect_link: str


@router.get("/all/{user_id}")
async def get_all_notifications_endpoint(user_id: int):
    return await get_all_notifications(user_id)

@router.post("/notify")
async def notify_user_endpoint(payload: CreateNotifDTO):
    return await notify_user(payload)

@router.delete("/read/{notification_id}")
async def read_notification_endpoint(notification_id: int):
    return await read_notification(notification_id)
