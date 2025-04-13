import os

from qdrant_client import QdrantClient

from constant import QDRANT_URL


def setup_client_qdrant():
    client_qdrant = QdrantClient(
        url=QDRANT_URL,
        port=6333,
        api_key=os.getenv("QDRANT_API_KEY"), )
    return client_qdrant
