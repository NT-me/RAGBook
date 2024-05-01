import os

from qdrant_client import QdrantClient


def setup_client_qdrant():
    client_qdrant = QdrantClient(
        url="https://ddffeb25-284c-447c-b81c-541719112ac9.us-east4-0.gcp.cloud.qdrant.io",
        port=6333,
        api_key=os.getenv("QDRANT_API_KEY"), )
    return client_qdrant
