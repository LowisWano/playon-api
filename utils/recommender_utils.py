import numpy as np
import pandas as pd
from datetime import datetime 
from geopy.distance import geodesic
from db.prisma import prisma
import logging
import json
from pathlib import Path
from sklearn.model_selection import train_test_split

# Logging Configs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model Storage Paths
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "recommender_model.pt"
TRAINING_METADATA_PATH = MODEL_DIR / "training_metadata.json"


"""
Utility Functions for the Recommender
"""
async def test_recommender():
    return "Recommender test!"


async def compute_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the geodesic distance between two geographic coordinates.

    Args:
        lat1 (float): Latitude of the first coordinate.
        lon1 (float): Longitude of the first coordinate.
        lat2 (float): Latitude of the second coordinate.
        lon2 (float): Longitude of the second coordinate.

    Returns:
        float: The distance in kilometers between the two coordinates.
               Returns a default distance of 120 if calculation fails.
    """
    try:
        return geodesic((lat1, lon1), (lat2, lon2)).km
    except:
        return 120
    

async def recommend_matches_by_preferences(user_id, max_distance=50):
    """
    Recommend matches for a user based on their preferred sports and location proximity.

    Parameters
    ----------
    user_id : int
        The ID of the user
    max_distance : int, optional
        Maximum distance in kilometers for match recommendations. Defaults to 50.

    Returns
    -------
    list
        List of recommended match records with their details and distances, or None if no matches are found
    """
    user = await prisma.users.find_unique(where={"id": user_id})
    if not user:
        return None
    
    matches = await prisma.match.find_many(
        where={
            "sport_id": {"in": user.preferred_sports},
            "match_start_date": {"gt": datetime.now()}
        }
    )

    recommended_matches = []
    for match in matches:
        match_dict = dict(match)
        if match_dict["location_lat"] and match_dict["location_long"]:
            distance = await compute_distance(
                user.location_lat,
                user.location_long,
                match_dict["location_lat"],
                match_dict["location_long"]
            )
            if distance <= max_distance:
                match_dict["distance"] = distance
                recommended_matches.append(match_dict)

    return recommended_matches if recommended_matches else None



async def recommend_matches_by_location(user_id, max_distance=50):
    """
    Recommend matches for a user based on location proximity.

    Parameters
    ----------
    user_id : int
        The ID of the user to recommend matches for.
    max_distance : int, optional
        The maximum distance in kilometers for location-based match recommendations. Defaults to 50.

    Returns
    -------
    list or None
        A list of recommended match records with their details and distance if found, otherwise None.
    """

    user = await prisma.users.find_unique(where={"id": user_id})
    if not user:
        return None

    matches = await prisma.match.find_many(where={"match_start_date": {"gt": datetime.now()}})

    recommended_matches = []
    for match in matches:
        match_dict = dict(match)
        if match_dict["location_lat"] and match_dict["location_long"]:
            distance = await compute_distance(
                user.location_lat,
                user.location_long,
                match_dict["location_lat"],
                match_dict["location_long"]
            )
            if distance <= max_distance:
                match_dict["distance"] = distance
                recommended_matches.append(match_dict)

    return recommended_matches if recommended_matches else None

