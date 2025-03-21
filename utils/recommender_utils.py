import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime, timezone
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


class CollaborativeFiltering(nn.Module):
    def __init__(self, num_users, num_matches, latent_dim=50):
        """
        Initialize a CollaborativeFiltering model.

        Parameters
        ----------
        num_users : int
            Number of users in the interaction matrix.
        num_matches : int
            Number of matches in the interaction matrix.
        latent_dim : int, optional
            Embedding dimension for users and matches. Defaults to 50.
        """
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, latent_dim)
        self.match_embedding = nn.Embedding(num_matches, latent_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.match_bias = nn.Embedding(num_matches, 1)

        # Initialize embeddings
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.match_embedding.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.match_bias.weight)

    def forward(self, user_ids, match_ids):
        """
        Compute the predicted interaction score for a set of user-match pairs.

        Parameters
        ----------
        user_ids : torch.tensor
            Tensor of user IDs.
        match_ids : torch.tensor
            Tensor of match IDs.

        Returns
        -------
        scores : torch.tensor
            Tensor of predicted interaction scores.
        """
        user_embeds = self.user_embedding(user_ids)
        match_embeds = self.match_embedding(match_ids)
        interaction = (user_embeds * match_embeds).sum(dim=1)
        user_b = self.user_bias(user_ids).squeeze()
        match_b = self.match_bias(match_ids).squeeze()
        return interaction + user_b + match_b


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


async def fetch_data():
    """
    Fetches user, match, match_user and sport data from the database.

    Returns
    -------
    dict
        Dictionary containing:
        - users: List of user records with nested user_info and user_location
        - matches: List of match records
        - match_users: List of match_user records
        - num_users: Total number of users
        - num_matches: Total number of matches 
        - sports: List of sport records
    """
    users = await prisma.users.find_many()
    matches = await prisma.match.find_many()
    match_users = await prisma.matchuser.find_many()
    sports = await prisma.sport.find_many()

    return {
        'users': users, 
        'matches': matches,
        'match_users': match_users,
        'num_users': len(users),
        'num_matches': len(matches),
        'sports': sports
        }


async def load_data():
    """
    Loads user, match and match_user data from the database and filters
    the records to only include past matches and the corresponding
    match-user interactions.

    Returns
    -------
    dict
        Dictionary containing:
        - users: List of user records with nested user_info and user_location
        - matches: List of match records for past matches
        - match_users: List of match_user records for past matches
    """
    data = await fetch_data()
    
    user_records = []
    match_records = []
    match_user_records = []

    # Create a dictionary of sports for easier lookup
    sports_dict = {sport.id: sport for sport in data['sports']}
    
    # Get current time in UTC (timezone-aware)
    current_time = datetime.now()

    for user in data['users']:
        user_records.append({
            'user_id': user.id,
            'first_name': user.first_name,
            'last_name': user.last_name,
            'location': user.location,
            'location_lat': user.location_lat,
            'location_long': user.location_long,
        })

    # Filter matches to only include past matches for training
    for match in data['matches']:
        # Convert match_start_date to naive datetime if it's timezone-aware
        match_start = match.match_start_date

        if match_start.tzinfo is not None:
            match_start = match_start.replace(tzinfo=None)
            
        if match_start < current_time:  # Only include past matches
            sport = sports_dict.get(match.sport_id)
            match_records.append({
                'match_id': match.id,
                'sport_id': match.sport_id,
                'sport_name': sport.sport_name if sport else "Unknown Sport",
                'location': match.location,
                'location_lat': match.location_lat,
                'location_long': match.location_long,
                'match_start_date': match_start,
                'match_end_date': match.match_end_date,
                'gender_preference': match.gender_preference,
                'skill_level': match.skill_level,
            })

    # Filter match_user records to only include those for past matches
    past_match_ids = {match['match_id'] for match in match_records}
    for match_user in data['match_users']:
        if match_user.match_id in past_match_ids:
            match_user_records.append({
                'match_user_id': match_user.id,
                'match_id': match_user.match_id,
                'user_id': match_user.user_id,
                'attended': match_user.attended,
                'is_host': match_user.is_host,
            })

    logger.info(f"Loaded {len(user_records)} users, {len(match_records)} past matches, and {len(match_user_records)} match-user interactions")
    
    return {
        'users': user_records,
        'matches': match_records,
        'match_users': match_user_records,
    }


async def create_user_match_interaction_matrix(val_split=0.2):
    """
    Creates a user-match interaction matrix for training and validation.

    Parameters
    ----------
    val_split : float, optional
        The proportion of data to use for validation, by default 0.2

    Returns
    -------
    tuple
        A tuple containing the following:
        - Y_train : The interaction matrix for training, with shape (num_users, num_matches)
        - R_train : The mask matrix for training, with shape (num_users, num_matches)
        - Y_val : The interaction matrix for validation, with shape (num_users, num_matches)
        - R_val : The mask matrix for validation, with shape (num_users, num_matches)
        - num_users : The total number of users
        - num_matches : The total number of matches
    """
    data = await load_data()

    users_df = pd.DataFrame(data['users'])
    matches_df = pd.DataFrame(data['matches'])
    match_user_df = pd.DataFrame(data['match_users'])

    unique_users = users_df['user_id'].unique()
    unique_matches = matches_df['match_id'].unique()

    num_users = len(unique_users)
    num_matches = len(unique_matches)

    user_to_index = {user_id: i for i, user_id in enumerate(unique_users)}
    match_to_index = {match_id: i for i, match_id in enumerate(unique_matches)}

    # Split train and validation sets
    train_df, val_df = train_test_split(match_user_df, test_size=val_split, random_state=42)

    def build_matrix(df):
        Y = np.zeros((num_users, num_matches))
        R = np.zeros((num_users, num_matches))
        for _, row in df.iterrows():
            u = user_to_index[row['user_id']]
            m = match_to_index[row['match_id']]
            Y[u, m] = 1 if row['attended'] else 0
            R[u, m] = 1
        return Y, R

    Y_train, R_train = build_matrix(train_df)
    Y_val, R_val = build_matrix(val_df)

    return Y_train, R_train, Y_val, R_val, num_users, num_matches


async def mean_norm(Y, R, num_users, num_matches):
    """
    Perform mean normalization on user ratings.

    Returns a new matrix Y_norm, where Y_norm[i, j] = Y[i, j] - mean(user i's ratings)
    """
    # Setting user means
    user_means = np.zeros(num_users)
    Y_norm = np.zeros_like(Y)

    # Calculate user means
    for i in range(num_users):
        played_matches = R[i, :]
        if np.sum(played_matches) > 0:
            user_means[i] = np.sum(Y[i, :] * played_matches) / np.sum(played_matches)

    # Mean Normalization
    for i in range(num_users):
        for j in range(num_matches):
            if R[i, j] == 1:  
                Y_norm[i, j] = Y[i, j] - user_means[i]

    return Y_norm


async def torch_preprocess():
    """
    Preprocess the user-match interaction matrix into PyTorch tensors.

    This function takes the output of `create_user_match_interaction_matrix` and
    converts it into PyTorch tensors after mean normalization. The output is a
    tuple of:

    - `Y_train_tensor`: The mean-normalized user-match interaction matrix for
      training.
    - `R_train_tensor`: The user-match interaction matrix for training.
    - `Y_val_tensor`: The mean-normalized user-match interaction matrix for
      validation.
    - `R_val_tensor`: The user-match interaction matrix for validation.
    - `num_users`: The number of users.
    - `num_matches`: The number of matches.

    Returns:
        tuple: A tuple of the preprocessed tensors and the number of users and
            matches.
    """
    Y_train, R_train, Y_val, R_val, num_users, num_matches = await create_user_match_interaction_matrix()

    def to_tensor(Y, R):
        Y_norm = np.zeros_like(Y)
        user_means = np.zeros(Y.shape[0])
        for i in range(Y.shape[0]):
            if np.sum(R[i]) > 0:
                user_means[i] = np.sum(Y[i] * R[i]) / np.sum(R[i])
        for i in range(Y.shape[0]):
            for j in range(Y.shape[1]):
                if R[i, j] == 1:
                    Y_norm[i, j] = Y[i, j] - user_means[i]
        return torch.tensor(Y_norm, dtype=torch.float32), torch.tensor(R, dtype=torch.float32)

    Y_train_tensor, R_train_tensor = to_tensor(Y_train, R_train)
    Y_val_tensor, R_val_tensor = to_tensor(Y_val, R_val)

    return Y_train_tensor, R_train_tensor, Y_val_tensor, R_val_tensor, num_users, num_matches


async def train_model(epochs=100, learning_rate=0.01, batch_size=64, early_stopping_patience=10):
    """
    Train the CollaborativeFiltering model.

    Parameters
    ----------
    epochs : int, optional
        The number of epochs to train the model, by default 100
    learning_rate : float, optional
        The learning rate for the Adam optimizer, by default 0.01
    batch_size : int, optional
        The batch size for training, by default 64
    early_stopping_patience : int, optional
        The number of epochs to wait before triggering early stopping, by default 10

    Returns
    -------
    dict
        A dictionary containing:

        - `model`: The trained model
        - `training_history`: A list of dictionaries containing the training and validation loss at each epoch
        - `best_model_state`: The best model state (i.e. the model state at the epoch with the lowest validation loss)
        - `final_loss`: The final validation loss
    """
    try:
        Y_train, R_train, Y_val, R_val, num_users, num_matches = await torch_preprocess()

        model = CollaborativeFiltering(num_users, num_matches)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        history = []
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        for epoch in range(epochs):
            model.train()
            total_loss = 0
            user_ids, match_ids = torch.where(R_train == 1)
            pairs = torch.stack([user_ids, match_ids], dim=1)
            indices = torch.randperm(pairs.size(0))

            for i in range(0, pairs.size(0), batch_size):
                batch_idx = indices[i:i+batch_size]
                batch = pairs[batch_idx]
                u = batch[:, 0]
                m = batch[:, 1]

                pred = model(u, m)
                targets = Y_train[u, m]

                loss = criterion(pred, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_train_loss = total_loss / (pairs.size(0) // batch_size)

            # Evaluate on validation set
            model.eval()
            with torch.no_grad():
                val_users, val_matches = torch.where(R_val == 1)
                val_preds = model(val_users, val_matches)
                val_targets = Y_val[val_users, val_matches]
                val_loss = criterion(val_preds, val_targets).item()

            history.append({
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_loss': val_loss
            })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")

        model.load_state_dict(best_model_state)
        return {
            'model': model,
            'training_history': history,
            'best_model_state': best_model_state,
            'final_loss': best_val_loss
        }

    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)
        raise


async def get_user_index(user_id):
    """
    Get the index of a user in the interaction matrix.
    
    Parameters
    ----------
    user_id : str
        The ID of the user
    
    Returns
    -------
    int
        The index of the user in the interaction matrix
    """
    data = await load_data()
    users_df = pd.DataFrame(data['users'])
    user_to_index = {user_id: i for i, user_id in enumerate(users_df['user_id'].unique())}
    
    if user_id not in user_to_index:
        raise ValueError(f"User {user_id} not found in the dataset")
        
    return user_to_index[user_id]


async def get_future_matches():
    """
    Get all future matches from the database.
    
    Returns
    -------
    list
        List of future match records
    """
    data = await fetch_data()
    current_time = datetime.now() 
    future_matches = []
    
    for match in data['matches']:
        # Handle both string and datetime objects
        match_start = match.match_start_date
        if match_start.tzinfo is not None:
            match_start = match_start.replace(tzinfo=None)

        if match_start > current_time:
            future_matches.append(match)
            
    return future_matches


async def get_recommended_matches(model, user_index, max_distance=50, top_n=5):
    """
    Get recommended matches for a user based on the trained model.
    
    Parameters
    ----------
    model : CollaborativeFiltering
        The trained model
    user_index : int
        The index of the user in the interaction matrix
    min_distance : float, optional
        Minimum distance in kilometers for match recommendations. Defaults to 50.
    top_n : int, optional
        Number of top recommendations to return. Defaults to 5.
    
    Returns
    -------
    list
        List of recommended match records with their predicted scores
    """
    try:
        # Get user data for distance filtering
        data = await load_data()
        users_df = pd.DataFrame(data['users'])
        
        # Get the actual user ID from the index
        user_id = users_df['user_id'].iloc[user_index]
        user_data = users_df[users_df['user_id'] == user_id].iloc[0]
        
        # Get future matches
        future_matches = await get_future_matches()
        
        if not future_matches:
            logger.info(f"No future matches found for user {user_id}")
            return []
        
        # Get the number of matches the model was trained on
        _, _, _, _, _, num_matches = await torch_preprocess()
        
        # Create tensors for prediction
        user_ids = torch.tensor([user_index] * len(future_matches))
        # Use match indices within the model's range
        match_indices = torch.arange(len(future_matches)) % num_matches
        
        # Get predictions
        model.eval()
        with torch.no_grad():
            predictions = model(user_ids, match_indices)
        
        # Create list of matches with predictions
        match_predictions = []
        for i, match in enumerate(future_matches):
            # Calculate distance
            distance = await compute_distance(
                user_data['location_lat'],
                user_data['location_long'],
                match.location_lat,
                match.location_long
            )
            
            # Only include matches within distance limit
            if distance <= max_distance:
                match_predictions.append({
                    'match': match,
                    'predicted_score': predictions[i].item(),
                    'distance': distance
                })
        
        # Sort by predicted score and get top N
        match_predictions.sort(key=lambda x: x['predicted_score'], reverse=True)
        top_recommendations = match_predictions[:top_n]
        
        logger.info(f"Generated {len(top_recommendations)} recommendations for user {user_id}")
        return top_recommendations
        
    except Exception as e:
        logger.error(f"Error getting recommendations: {str(e)}", exc_info=True)
        raise


async def recommend_matches(user_id, max_distance=50, top_n=5):
    """
    Get match recommendations for a user.
    
    Parameters
    ----------
    user_id : str
        The ID of the user
    min_distance : float, optional
        Minimum distance in kilometers for match recommendations. Defaults to 50.
    top_n : int, optional
        Number of top recommendations to return. Defaults to 5.
    
    Returns
    -------
    list
        List of recommended match records with their predicted scores and distances
    """
    try:
        # Get or train model
        model_result = await get_or_train_model()
        model = model_result['model']
        
        # Get user index
        user_index = await get_user_index(user_id)
        
        # Get recommended matches
        recommendations = await get_recommended_matches(
            model=model,
            user_index=user_index,
            max_distance=max_distance,
            top_n=top_n
        )
        
        # Get sports data for lookup
        data = await fetch_data()
        sports_dict = {sport.id: sport for sport in data['sports']}
        
        # Format recommendations for response
        formatted_recommendations = []
        for rec in recommendations:
            match = rec['match']
            sport = sports_dict.get(match.sport_id)
            formatted_recommendations.append({
                'match_id': match.id,
                'sport_name': sport.sport_name if sport else "Unknown Sport",
                'location': match.location,
                'match_start_date': match.match_start_date,
                'predicted_score': rec['predicted_score'],
                'distance_km': rec['distance']
            })
        
        return formatted_recommendations
        
    except Exception as e:
        logger.error(f"Error in recommend_matches: {str(e)}", exc_info=True)
        raise


async def save_model(model, training_history, metadata):
    """
    Save the trained model and training metadata to disk.
    
    Parameters
    ----------
    model : CollaborativeFiltering
        The trained model to save
    training_history : list
        List of training metrics per epoch
    metadata : dict
        Dictionary containing training metadata (e.g., timestamp, parameters)
    """
    try:
        # Save model state
        torch.save(model.state_dict(), MODEL_PATH)
        
        # Save training metadata
        metadata['last_training_date'] = datetime.now().isoformat()
        metadata['training_history'] = training_history
        
        with open(TRAINING_METADATA_PATH, 'w') as f:
            json.dump(metadata, f, indent=4)
            
        logger.info(f"Model and metadata saved successfully to {MODEL_DIR}")
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}", exc_info=True)
        raise


async def load_model(num_users, num_matches):
    """
    Load the saved model and metadata from disk.
    
    Parameters
    ----------
    num_users : int
        Number of users in the current dataset
    num_matches : int
        Number of matches in the current dataset
    
    Returns
    -------
    tuple
        (model, metadata) if model exists, (None, None) otherwise
    """
    try:
        if not MODEL_PATH.exists() or not TRAINING_METADATA_PATH.exists():
            logger.info("No saved model found")
            return None, None
            
        # Load model
        model = CollaborativeFiltering(num_users, num_matches)
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()
        
        # Load metadata
        with open(TRAINING_METADATA_PATH, 'r') as f:
            metadata = json.load(f)
            
        logger.info(f"Model loaded successfully from {MODEL_PATH}")
        return model, metadata
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}", exc_info=True)
        return None, None


async def should_retrain():
    """
    Check if the model should be retrained based on the last training date.
    
    If the model has never been trained or more than 7 days have passed since the last training,
    return True. Otherwise, return False.
    
    Returns
    -------
    bool
        Whether the model should be retrained
    """
    try:
        if not TRAINING_METADATA_PATH.exists():
            return True
            
        with open(TRAINING_METADATA_PATH, 'r') as f:
            metadata = json.load(f)
            
        last_training_date = datetime.fromisoformat(metadata['last_training_date'])
        days_since_training = (datetime.now() - last_training_date).days
        
        # Retrain if more than 7 days have passed
        return days_since_training >= 7
    except Exception as e:
        logger.error(f"Error checking retrain condition: {str(e)}", exc_info=True)
        return True


async def get_or_train_model(epochs=100, learning_rate=0.001, batch_size=64, early_stopping_patience=10):
    """
    Get or train the CollaborativeFiltering model.

    This function checks if the model needs to be retrained based on the last
    training date. If retraining is not required, it attempts to load an existing
    model. Otherwise, it trains a new model and saves it along with the training
    metadata.

    Parameters
    ----------
    epochs : int, optional
        The number of epochs to train the model, by default 100
    learning_rate : float, optional
        The learning rate for the Adam optimizer, by default 0.001
    batch_size : int, optional
        The batch size for training, by default 64
    early_stopping_patience : int, optional
        The number of epochs to wait before triggering early stopping, by default 10

    Returns
    -------
    dict
        A dictionary containing:
        - `model`: The trained model
        - `training_history`: A list of dictionaries containing the training and validation loss at each epoch
        - `best_model_state`: The best model state (i.e., the model state at the epoch with the lowest validation loss)
        - `final_loss`: The final validation loss
    """

    try:
        # Get current dataset dimensions
        _, _, _, _, num_users, num_matches  = await torch_preprocess()
        
        # Check if we should retrain
        if not await should_retrain():
            # Try to load existing model
            model, metadata = await load_model(num_users, num_matches)
            if model is not None:
                logger.info("Using existing model")
                return {
                    'model': model,
                    'training_history': metadata['training_history'],
                    'best_model_state': model.state_dict(),
                    'final_loss': metadata['training_history'][-1]['val_loss']
                }
        
        # Train new model
        logger.info("Training new model")
        training_result = await train_model(
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            early_stopping_patience=early_stopping_patience
        )
        
        # Save the new model
        await save_model(
            model=training_result['model'],
            training_history=training_result['training_history'],
            metadata={
                'num_users': num_users,
                'num_matches': num_matches,
                'training_params': {
                    'epochs': epochs,
                    'learning_rate': learning_rate,
                    'batch_size': batch_size,
                    'early_stopping_patience': early_stopping_patience
                }
            }
        )
        
        return training_result
        
    except Exception as e:
        logger.error(f"Error in get_or_train_model: {str(e)}", exc_info=True)
        raise

