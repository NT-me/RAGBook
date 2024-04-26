import os
import re
import string
from typing import List
from qdrant_client import QdrantClient
from qdrant_client.http.models import models
from sentence_transformers import SentenceTransformer
import uuid
import csv

sentences = ["This is an example sentence", "Each sentence is converted"]

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

    for sentence in sentences:
        embed_sentence = MODEL.encode(sentence, show_progress_bar=True)
        ret.append((sentence, embed_sentence.reshape(embed_sentence.shape)))
    return ret


def upsert_points(vec, orginal_text, clean_text, orignal_indice):
    client = QdrantClient(
        url="https://ddffeb25-284c-447c-b81c-541719112ac9.us-east4-0.gcp.cloud.qdrant.io",
        port=6333,
        api_key=os.getenv("QDRANT_API_KEY"), )

    points = []
    for emb in vec:
        model = models.PointStruct(
            id=str(uuid.uuid4()),
            payload={
                "title": "The cry of nature; or, an appeal to mercy and to justice, on behalf of the persecuted animals.".capitalize(),
                "orginal_text": orginal_text,
                "clean_text": clean_text,
                "original_indice": orignal_indice
            },
            vector=emb[1],
        )
        points.append(model)

    client.upsert(
        collection_name="contents_text",
        points=points,
    )


if __name__ == "__main__":
    with open("./data/output2.csv", 'r', encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["orginal_batch10"]:
                sentences = convert_content_to_sentences(row["corrected_text"])
                emb_vec = vectorize_sentences(sentences)
                upsert_points(
                    vec=emb_vec,
                    orginal_text=row["orginal_batch10"],
                    clean_text=row["corrected_text"],
                    orignal_indice=row["orginal_indice"]
                )
