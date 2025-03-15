from db.prisma import prisma
from utils.http_error import BadRequestErrorResponse


async def get_team_matches(team_id: int):
    return await prisma.teammatch.find_many(
        where={
            "team_id": team_id
        }
    )
    
async def get_team_match_by_id(match_id: int):
    return await prisma.teammatch.find_unique(
        where={
            "id": match_id
        }
    )
    
async def create_team_match():
    try:

        # person who is assigned to implement match creation/recommendation
        # should also implement the logic here

        return
    except Exception as e:
        raise BadRequestErrorResponse(error_message=str(e)).as_exception()
    

async def remove_team_match(team_match_id: int):
    try:
        return await prisma.teammatch.delete(
            where={
                "id": team_match_id
            }
        )
    except Exception as e:
        raise BadRequestErrorResponse(error_message=str(e)).as_exception()