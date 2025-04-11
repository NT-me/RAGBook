# This example requires the 'message_content' intent.
import json
import os

import discord
import requests
from discord import Embed
from dotenv import load_dotenv
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import models

from constant import QDRANT_URL, JINAI_URL, JINAI_HEADERS

load_dotenv()  # take environment variables from .env.

intents = discord.Intents.default()
intents.message_content = True
system_message = open("prompt/client_discord/system.txt", "r", encoding="utf-8").read()

client = discord.Client(intents=intents)

client_qdrant = QdrantClient(
    url=QDRANT_URL,
    port=6333,
    api_key=os.getenv("QDRANT_API_KEY"), )

api_key = os.environ["MISTRAL_API_KEY"]
model = "open-mixtral-8x22b"

client_mistral = MistralClient(api_key=api_key)


def get_query_emb(query: str):
    data = {
        "model": "jina-clip-v2",
        "task": "retrieval.query",
        "input": [
            {"text": query},
        ]
    }
    response = requests.post(JINAI_URL, headers=JINAI_HEADERS, data=json.dumps(data))
    ret = response.json()["data"][0]["embedding"]
    return ret

@client.event
async def on_ready():
    print(f'We have logged in as {client.user}')


@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content.startswith('$rag'):
        message_without_command = message.content.replace("$rag", "")
        embed_msg = get_query_emb(message_without_command)

        template_message = open("prompt/client_discord/user.txt", "r", encoding="utf-8").read()

        qdrant_search = client_qdrant.search(
            collection_name="contents_text",
            limit=10,
            query_vector=embed_msg
        )

        filtered_qs = [x for x in qdrant_search if x.score > 0.5]

        sources = []
        for i, point in enumerate(filtered_qs):
            e = point.payload
            local_e = Embed(
                title=f"Source #{i} - {e['title']}",
                description=f"""
                Score {point.score}
                """
            )
            max_line = (int(e["original_indice"]) * 10) + 10
            local_e.add_field(
                name="Approximative line number",
                value=f"{int(e['original_indice']) * 10} - {max_line}"
            )
            local_e.add_field(
                name="Texte propre",
                value=e["clean_text"][:50] + "..."
            )
            local_e.add_field(
                name="Texte original",
                value=e["orginal_text"][:50] + "..."
            )
            sources.append(local_e)

        clean_texts = [str({"clean_text": x.payload.get("clean_text"), "title of book": x.payload.get("title")}) for x
                       in filtered_qs]
        template_message = template_message.replace("{retrieved_chunk}", "\n".join(clean_texts))
        question_message = template_message.replace("{question}", message_without_command)

        chat_response = client_mistral.chat(
            model=model,
            messages=[ChatMessage(role="system", content=system_message),
                      ChatMessage(role="user", content=question_message)],
            temperature=0
        )

        resp = chat_response.choices[0].message.content
        await message.channel.send(resp, embeds=sources[:10])


if __name__ == "__main__":
    client.run(os.environ["DISCORD_API_KEY"])
