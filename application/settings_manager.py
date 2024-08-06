import json
import os

import redis 
from dotenv import load_dotenv
import redis.client

# Path to the .env file
dotenv_path = os.path.join(os.path.dirname(__file__), "../.env")

# Clear the environment variables
os.environ.clear()

# Load environment variables from the .env file
load_dotenv(dotenv_path)

redis_client = redis.Redis.from_url(os.environ.get("REDIS_URL"), decode_responses=True)

def insert_settings(app_id: str, new_settings: dict):

    # Convert settings dictionary to JSON string
    setttings_json= json.dumps(new_settings)

    # Insert JSON string in Reddis with app_id as the key
    redis_client.set(app_id, setttings_json)

def fetch_settings(app_id: str) -> dict:
    settings= redis_client.get(app_id)
    if settings:
        return json.loads(settings)
    return {}


