# This example requires the 'message_content' intent.
import os
from zlib import adler32

import discord
from discord import Embed, Message
from dotenv import load_dotenv
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import models

load_dotenv()  # take environment variables from .env.

intents = discord.Intents.default()
intents.message_content = True
system_message = open("prompt/client_discord/system.txt", "r", encoding="utf-8").read()

client = discord.Client(intents=intents)
MODEL = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

client_qdrant = QdrantClient(
    url="https://ddffeb25-284c-447c-b81c-541719112ac9.us-east4-0.gcp.cloud.qdrant.io",
    port=6333,
    api_key=os.getenv("QDRANT_API_KEY"), )

api_key = os.environ["MISTRAL_API_KEY"]
model = "open-mixtral-8x22b"

client_mistral = MistralClient(api_key=api_key)


@client.event
async def on_ready():
    print(f'We have logged in as {client.user}')


@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content.startswith('$rag'):
        async with message.channel.typing():
            message_without_command = message.content.replace("$rag", "")
            embed_msg = MODEL.encode(message_without_command, show_progress_bar=True)

            template_message = open("prompt/client_discord/user.txt", "r", encoding="utf-8").read()

            qdrant_search = client_qdrant.search(
                collection_name="contents_text",
                limit=50,
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
                    """,
                    color=adler32(e['title'].encode("utf-8")) % 16777215
                )
                max_line = (int(e["original_indice"]) * 10) + 10
                local_e.add_field(
                    name="Approximative line number",
                    value=f"{int(e["original_indice"]) * 10} - {max_line}"
                )
                local_e.add_field(
                    name="Texte propre",
                    value=e["clean_text"][:200] + "..."
                )
                local_e.add_field(
                    name="Texte original",
                    value=e["orginal_text"][:200] + "..."
                )
                sources.append(local_e)

            clean_texts = [str({"clean_text": x.payload.get("clean_text"), "title of book": x.payload.get("title")}) for
                           x
                           in filtered_qs]
            template_message = template_message.replace("{retrieved_chunk}", "\n".join(clean_texts))
            question_message = template_message.replace("{question}", message_without_command)

            # With streaming
            stream_response = client_mistral.chat_stream(
                model=model,
                messages=[
                    ChatMessage(role="system", content=system_message),
                    ChatMessage(role="user", content=question_message)
                ],
                temperature=0.5)

            accu_resp = []
            for chunk in stream_response:
                content = chunk.choices[0].delta.content
                if "\n" in content or len("".join(accu_resp)) >= 1500:
                    await message.channel.send("".join(accu_resp))
                    accu_resp.clear()
                accu_resp.append(content)
            await message.channel.send("".join(accu_resp))
            thread_msg = await message.channel.send("Thread pour les sources")

            source_thread = await message.channel.create_thread(
                name="Source Thread",
                message=thread_msg
            )

            for source in sources:
                await source_thread.send(embed=source)


if __name__ == "__main__":
    client.run(os.environ["DISCORD_API_KEY"])
