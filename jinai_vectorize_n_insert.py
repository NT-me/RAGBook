import os
import re
import uuid
from datetime import datetime
from typing import List

import requests
import spacy
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_openapi_client.models import models

from constant import JINAI_URL, JINAI_HEADERS, QDRANT_URL

load_dotenv()

en_nlp = nlp = spacy.load("en_core_web_sm")


def _split_by_word_respecting_sent_boundary(
        document_content: str,
) -> List[str]:
    """
    Splits the text into slices of slice_length words while respecting sentence boundaries.

    :param document_content: The text to split into slices
    :param lang: The language of the text in format 'en', 'fr', 'es', etc.
    :param slice_length:  The maximum number of words in a slice
    :return: Slices of text with a maximum of slice_length words
    """
    text = re.sub(" +", " ", re.sub(r"\n+", " ", document_content)).strip()

    nlp_model = en_nlp
    spacy_doc = nlp_model(text)
    ret = []

    for s in spacy_doc.sents:
        ret.append(
            s.text.strip()
        )

    return ret


def vectorize_sentences(
        sentences: List[str]
):
    ret = []

    batches = []
    batch_size = 50
    for i in range(0, len(sentences), batch_size):
        batches.append(
            sentences[i:i + batch_size]
        )

    for i, batch in enumerate(batches):
        print(f"{i}/{len(batches)}")
        texts = [{"text": s} for s in batch]
        data = {
            "model": "jina-clip-v2",
            "input": texts

        }
        response = requests.post(JINAI_URL, headers=JINAI_HEADERS, json=data)
        response.raise_for_status()
        embeddings = response.json().get("data")

        for ret_obj in embeddings:
            embedding = ret_obj["embedding"]
            index = ret_obj["index"]
            ret.append((batch[index], embedding))
    return ret


def upsert_points(vec, orginal_text, clean_text, orignal_indice):
    client = QdrantClient(
        url=QDRANT_URL,
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
        sentences: List[str] = _split_by_word_respecting_sent_boundary(document_content=f.read())
        print(f"There is {len(sentences)} sentences")
        sentences_and_embs = vectorize_sentences(sentences=sentences)

        i = 0
        for s, e in sentences_and_embs:
            with open(f"./data/{datetime.now().timestamp()}_embs_backups.csv", 'w') as fin:
                fin.write(f'"{s}", {e}\n')
            upsert_points(vec=e, orginal_text=s, clean_text=s, orignal_indice=i)
            i += 1
            print(f"{i} / {len(sentences_and_embs)}")


if __name__ == "__main__":
    main()
