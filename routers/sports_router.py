from fastapi import APIRouter
from services.sports_services import *

router = APIRouter(
    prefix="/sports",
    tags=["sports"]    
)


@router.get("/all")
async def get_all_sports_endpoint():
    return await get_all_sports()


@router.get("/get-sport/{sport_id}")
async def get_sport_by_id_endpoint(sport_id: int):
    return await get_sport_by_id(sport_id)


@router.get("/get-sport-by-name/{sport_name}")
async def get_sport_by_name_endpoint(sport_name: str):
    return await get_sport_by_name(sport_name)