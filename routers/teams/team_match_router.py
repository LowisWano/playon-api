from fastapi import APIRouter
from services.teams.team_match_services import *

router = APIRouter(
    prefix="/team-match",
    tags=["team-match"]    
)

@router.get("/{team_id}")
async def get_team_match_endpoint(team_id: int):
    return await get_team_matches(team_id)

@router.get("/team-match/{match_id}")
async def get_team_match_by_id_endpoint(match_id: int):
    return await get_team_match_by_id(match_id)

# @router.post("/create-match")
# async def create_team_match_endpoint(payload: CreateTeamMatchDTO):
    # return await create_team_match(payload)

@router.delete("/delete-match/{team_match_id}")
async def remove_team_match_endpoint(team_match_id: int):
    return await remove_team_match(team_match_id)
