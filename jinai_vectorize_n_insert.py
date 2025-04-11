import os
import re
import string
import uuid
from typing import List
from dotenv import load_dotenv
import requests
from qdrant_client import QdrantClient
from qdrant_openapi_client.models import models

load_dotenv()

JINAI_URL = 'https://api.jina.ai/v1/embeddings'
HEADERS = {
    'Content-Type': 'application/json',
    'Authorization': os.getenv("JINAI_API_KEY")
}


def convert_content_to_sentences(content: str):
    sentences = []
    splitted_content = content.split(' ')

    words = []
    for i, word in enumerate(splitted_content):
        if i % 8000 == 0 or word.endswith('.'):
            sentence = " ".join(words)
            sentence.strip()
            sentences.append(sentence)
            words.clear()

        if word != '':
            words.append(word)
    return sentences


def vectorize_sentences(
        sentences: List[str]
):
    ret = []

    for i, sentence in enumerate(sentences):
        print(i)

        data = {
            "model": "jina-clip-v2",
            "input": [
                {
                    "text": sentence
                }
            ]
        }
        response = requests.post(JINAI_URL, headers=HEADERS, json=data)
        response.raise_for_status()
        embeddings = response.json().get("data")[0]["embedding"]
        ret.append((sentence, embeddings))
    return ret


def upsert_points(vec, orginal_text, clean_text, orignal_indice):
    client = QdrantClient(
        url="https://edb8d98e-72dd-4739-bcb3-c18f368497d5.eu-central-1-0.aws.cloud.qdrant.io",
        port=6333,
        api_key=os.getenv("QDRANT_API_KEY"), )

    client.upsert(
        collection_name="contents_text",
        points=[models.PointStruct(
            id=str(uuid.uuid4()),
            payload={
                "title": "The Cry of Nature".capitalize(),
                "orginal_text": orginal_text,
                "clean_text": clean_text,
                "original_indice": orignal_indice
            },
            vector=vec,
        )],
    )


def main():
    with open("./data/cry_of_the_nature_wikisource.md", 'r', encoding="utf-8") as f:
        sentences: List[str] = convert_content_to_sentences(f.read())
        sentences_and_embs = vectorize_sentences(sentences=sentences)

        i = 0
        for s, e in sentences_and_embs:
            with open("./data/embs_backups.csv", 'w') as fin:
                fin.write(f'"{s}", {e}\n')
            upsert_points(vec=e, orginal_text=s, clean_text=s, orignal_indice=i)
            i += 1


if __name__ == "__main__":
    main()
