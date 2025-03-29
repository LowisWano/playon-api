from pydantic import BaseModel

class CreateGroupChatDTO(BaseModel):
    created_by: int
    gc_name: str
    image: str
    members: list[int] = []
    
class AddToGroupChatDTO(BaseModel):
    user_id: int
    group_id: int
    isAdmin: bool
    isAutoApprove: bool

class SendMessageDTO(BaseModel):
    room_id: int
    sender_id: int
    sender_name: str
    content: str
    
class CreateConversationDTO(BaseModel):
    sender_id: int
    receiver_id: int
    room_id: int
    content: str
