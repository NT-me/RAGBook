import os

from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.


JINAI_URL = 'https://api.jina.ai/v1/embeddings'
JINAI_HEADERS = {
    'Content-Type': 'application/json',
    'Authorization': os.getenv("JINAI_API_KEY")
}
QDRANT_URL = "https://edb8d98e-72dd-4739-bcb3-c18f368497d5.eu-central-1-0.aws.cloud.qdrant.io"
