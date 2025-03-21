from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from utils.recommender_utils import *

router = APIRouter(
    prefix="/recommender",
    tags=["recommender"]
)


"""
Utility Functions for the Recommender health check
"""
@router.get("/test")
async def test():
    return await test_recommender()


@router.get("/test-fetch")
async def test_fetch():
    result = await fetch_data()
    return {"result": result}


@router.get("/test-preprocess")
async def test_fetch():
    result = await load_data()
    return {"result": result}


"""
Collaborative Filtering Based Recommendation Endpoint
"""
@router.get("/get-recommendations/{user_id}")
async def get_recommendations(user_id: int, max_distance: int = 50):
    """
    Get personalized match recommendations for a use, based on the history of interactions.
    
    Parameters:
    - user_id: The ID of the user
    - max_distance: Maximum distance in kilometers for match recommendations (default: 50)
    
    Returns:
    - List of recommended matches with their details and predicted scores
    """
    try:
        recommendations = await recommend_matches(
            user_id=user_id,
            max_distance=max_distance,
            top_n=8,
        )
        
        if not recommendations:
            raise HTTPException(
                status_code=404,
                detail="No recommendations found for this user"
            )
            
        return recommendations
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
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
