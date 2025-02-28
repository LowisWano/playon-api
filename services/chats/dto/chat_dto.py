from pydantic import BaseModel

class CreateGroupChatDTO(BaseModel):
    created_by: int
    title: str

class SendMessageDTO(BaseModel):
    chat_id: int
    sender_id: int
    content: str
    
class CreateConversationDTO(BaseModel):
    sender_id: int
    receiver_id: int
    room_id: int
