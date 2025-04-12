import os
import re
import string
from time import sleep
from typing import List

from dotenv import load_dotenv
from numpy import ndarray
from qdrant_client import QdrantClient
from qdrant_client.http.models import models
from sentence_transformers import SentenceTransformer
import uuid
import csv

from torch import Tensor

MODEL = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')


def _remove_punctuation(input_string):
    # Cette expression régulière correspond à tous les caractères de ponctuation
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    # Supprime tous les caractères de ponctuation de la chaîne d'entrée
    output_string = regex.sub('', input_string)
    return output_string


def convert_content_to_sentences(content: str):
    sentences = []
    splitted_content = content.split(' ')

    words = []
    for i, word in enumerate(splitted_content):
        if i % MODEL.max_seq_length == 0 or word.endswith('.'):
            sentence = " ".join(words)
            sentence.strip()
            sentences.append(_remove_punctuation(sentence))
            words.clear()

        if word != '':
            words.append(word)
    return sentences


def vectorize_sentences(
        sentences: List[str]
):
    ret = []

    for i, sentence in enumerate(sentences):
        embed_sentence = MODEL.encode(sentence, show_progress_bar=True, device="cpu")
        upsert_points(
            vec=[(sentence, embed_sentence.reshape(embed_sentence.shape))],
            orginal_text="",
            clean_text="",
            orignal_indice=i
        )
    return ret


def upsert_points(vec: list[tuple[str, ndarray | Tensor]], orginal_text, clean_text, orignal_indice):
    client = QdrantClient(
        url="https://ddffeb25-284c-447c-b81c-541719112ac9.us-east4-0.gcp.cloud.qdrant.io",
        port=6333,
        api_key=os.getenv("QDRANT_API_KEY"), )

    points = []
    for emb in vec:
        model = models.PointStruct(
            id=str(uuid.uuid4()),
            payload={
                "title": "La Chronique du mois".capitalize(),
                "orginal_text": emb[0],
                "clean_text": emb[0],
                "original_indice": orignal_indice
            },
            vector=emb[1],
        )
        points.append(model)
    for i in range(0, 10):
        try:
            client.upsert(
                collection_name="contents_text",
                points=points,
                wait=False
            )
        except Exception as e:
            print(f"Exception occurred: {str(e)}")
            sleep(i * 2)
        else:
            break


def main():
    with open("./data/lachroniquedumo00clavgoog_djvu.txt", 'r', encoding="utf-8") as f:
        sentences = convert_content_to_sentences(f.read())
        emb_vec: list[tuple[str, ndarray | Tensor]] = vectorize_sentences(sentences)


if __name__ == "__main__":
    load_dotenv()  # take environment variables from .env
    main()
