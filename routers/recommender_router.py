from fastapi import APIRouter, HTTPException
from utils.recommender_utils import *

router = APIRouter(
    prefix="/recommender",
    tags=["recommender"]
)

"""
Simple Recommender Endpoints
"""
@router.get("/get-recommendations-by-preferences/{user_id}")
async def get_recommendations_by_preferences(user_id: int, max_distance: int = 50):
    """
    Get personalized match recommendations for a user based on their preferred sports.
    
    Parameters:
    - user_id: The ID of the user
    - max_distance: Maximum distance in kilometers for match recommendations (default: 50)
    
    Returns:
    - List of recommended matches with their details
    """
    return await recommend_matches_by_preferences(user_id, max_distance)


@router.get("/get-recommendations-by-location/{user_id}")
async def get_recommendations_by_location(user_id: int, max_distance: int = 50):
    """
    Get match recommendations for a user based on their location.
    
    Parameters:
    - user_id: The ID of the user
    - max_distance: Maximum distance in kilometers for match recommendations (default: 50)
    
    Returns:
    - List of recommended matches with their details
    """
    return await recommend_matches_by_location(user_id, max_distance)
