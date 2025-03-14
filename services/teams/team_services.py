from db.prisma import prisma
from utils.http_error import BadRequestErrorResponse
from pydantic import BaseModel
from typing import Optional
from services.notification_services import notify_user, CreateNotifDTO

class CreateTeamDTO(BaseModel):
    team_name: str
    coach_id: Optional[int]
    creator_id: int
    
    
async def get_all_teams():
    return await prisma.teams.find_many()

async def get_all_teams_of_user(user_id: int):
    return await prisma.teamuser.find_many(
        where={
            "user_id": user_id
        },
        include={
            "teams": True
        }
    )
    
async def get_all_team_members(team_id: int):
    return await prisma.team.find_first(
        where={
            "id": team_id
        },
        include={
            "team_users": True
        }
    )
    
async def get_team_by_id(team_id: int):
    return await prisma.team.find_first(
        where={
            "id": team_id
        }
    )
    
async def create_team(payload: CreateTeamDTO):
    
    try:
        name_exists = await prisma.team.find_first(
            where={
                "team_name": payload.team_name
            }
        )
        
        if name_exists:
            raise BadRequestErrorResponse(error_message="Team name already exists").as_exception()
        
        return await prisma.team.create(
            data={
                "team_name": payload.team_name,
                "coach_id": payload.coach_id if payload.coach_id else None,
                "leader_id": payload.creator_id,
            },
        )
    except Exception as e:
        raise BadRequestErrorResponse(error_message=str(e)).as_exception()
    
    
async def add_user_to_team(team_id: int, user_id: int):
    try:
        name_exists = await prisma.teamuser.find_first(
            where={
                "team_id": team_id,
                "user_id": user_id
            }
        )
        
        if name_exists:
            raise BadRequestErrorResponse(error_message="User already exists in the team.").as_exception()
        
        await notify_user(
            CreateNotifDTO(
                notif_from_id=team_id,
                notif_to_id=user_id,
                message=f"You have been added to a team.",
                notif_type="Team",
                redirect_link=None
            )
        )
        
        return await prisma.teamuser.create(
            data={
                "team_id": team_id,
                "user_id": user_id
            }
        )
    except Exception as e:
        raise BadRequestErrorResponse(error_message=str(e)).as_exception()
    

async def update_team_name(team_id: int, new_team_name: str):
    try:
        return await prisma.team.update(
            where={
                "id": team_id
            },
            data={
                "team_name": new_team_name
            }
        )
    except Exception as e:
        raise BadRequestErrorResponse(error_message=str(e)).as_exception()


async def remove_user_from_team(team_id: int, user_id: int):
    try:
        return await prisma.teamuser.delete(
            where={
                "team_id": team_id,
                "user_id": user_id
            }
        )
    except Exception as e:
        raise BadRequestErrorResponse(error_message=str(e)).as_exception()
    

async def disband_team(team_id: int, team_name: str):
    try:
        
        users = await prisma.teamuser.find_many(
            where={
                "team_id": team_id
            }
        )
        
        for user in users:
            await notify_user(
                CreateNotifDTO(
                    notif_from_id=None,
                    notif_to_id=user.user_id,
                    message=f"{team_name} has been disband.",
                    notif_type="Team",
                    redirect_link=None
                )
            )  
        await prisma.teamuser.delete_many(
            where={
                "team_id": team_id
            }
        )
        await prisma.team.delete(
            where={
                "id": team_id
            }
        )
    except Exception as e:
        raise BadRequestErrorResponse(error_message=str(e)).as_exception()