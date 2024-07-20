import os
import requests
from dotenv import load_dotenv

# load_dotenv()

# access_token = os.environ['API_TOKEN']
API_URL = "https://api-inference.huggingface.co/models/Dmyadav2001/Sentimental-Analysis"
headers = {"Authorization": "Bearer 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"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
	
output = query({
	"inputs": """I love you like i fucking hate you"""
})

print(output)