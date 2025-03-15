from fastapi import APIRouter
from services.teams.team_services import *

router = APIRouter(
    prefix="/team",
    tags=["team"]    
)


@router.get("/all")
async def get_all_teams_endpoint():
    return await get_all_teams()

@router.get("/all-teams-of-user/{user_id}")
async def get_all_teams_of_user_endpoint(user_id: int):
    return await get_all_teams_of_user(user_id)

@router.get("/members/{team_id}")
async def get_all_team_members_endpoint(team_id: int):
    return await get_all_team_members(team_id)

@router.get("/get-team/{team_id}")
async def get_team_by_id_endpoint(team_id: int):
    return await get_team_by_id(team_id)

@router.post("/create-team")
async def create_team_endpoint(payload: CreateTeamDTO):
    return await create_team(payload)

@router.post("/add-user-to-team")
async def add_user_to_team_endpoint(team_id: int, user_id: int):
    return await add_user_to_team(team_id, user_id)

@router.patch("/update-team")
async def update_team_endpoint(team_id: int, team_name: str):
    return await update_team_name(team_id, team_name)

@router.delete("/remove-user-from-team")
async def remove_user_from_team_endpoint(team_id: int, user_id: int):
    return await remove_user_from_team(team_id, user_id)

@router.delete("/disband-team")
async def disband_team_endpoint(team_id: int):
    return await disband_team(team_id)