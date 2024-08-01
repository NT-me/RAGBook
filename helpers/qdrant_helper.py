import os

from qdrant_client import QdrantClient


def setup_client_qdrant():
    client_qdrant = QdrantClient(
        url=os.getenv("QDRANT_HOST"),
        port=6333,
        api_key=os.getenv("QDRANT_API_KEY"), )
    return client_qdrant
