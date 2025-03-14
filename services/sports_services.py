from db.prisma import prisma

async def get_all_sports():
    return await prisma.sport.find_many()


async def get_sport_by_id(sport_id: int):
    return await prisma.sport.find_unique(
        where={
            "id": sport_id
        }
    )
    
async def get_sport_by_name(sport_name: str):
    return await prisma.sport.find_unique(
        where={
            "sport_name": sport_name
        }
    )