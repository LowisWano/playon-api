from pydantic import BaseModel

class CreateGroupChatDTO(BaseModel):
    created_by: int
    title: str

class CreateGroupUsersDTO(BaseModel):
    user_id: int
    class Config:
        orm_mode = True