from pydantic import BaseModel

class CreateGroupChatDTO(BaseModel):
    created_by: int
    gc_name: str
    image: str
    
class AddToGroupChatDTO(BaseModel):
    user_id: int
    group_id: int
    isAdmin: bool

class SendMessageDTO(BaseModel):
    room_id: int
    sender_id: int
    content: str
    
class CreateConversationDTO(BaseModel):
    sender_id: int
    receiver_id: int
    room_id: int
