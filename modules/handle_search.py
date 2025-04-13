import os
from statistics import mean
from zlib import adler32

import yaml
from discord import Embed
from discord.ext.commands import Context, Cog, command
from dotenv import load_dotenv
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from sentence_transformers import SentenceTransformer
from qdrant_client.http.models import models
import requests
from helpers.qdrant_helper import setup_client_qdrant
from constant import JINAI_URL, JINAI_HEADERS

load_dotenv()  # take environment variables from .env.

system_message = open("prompt/client_discord/system.txt", "r", encoding="utf-8").read()
template_user_message = open("prompt/client_discord/user.txt", "r", encoding="utf-8").read()

client_qdrant = setup_client_qdrant()

mistral_api_key = os.environ["MISTRAL_API_KEY"]
model = "open-mixtral-8x22b"

client_mistral = MistralClient(api_key=mistral_api_key)
yaml_key = "Textes actifs"


def vectorize_msg(msg_to_vecto: str):
    data = {
        "model": "jina-clip-v2",
        "task": "retrieval.query",
        "input": [
            {"text": msg_to_vecto}
        ]
    }
    response = requests.post(JINAI_URL, headers=JINAI_HEADERS, json=data)
    return response.json().get("data")[0].get("embedding")


class HandleSearch(Cog):
    def __init__(self, bot):
        self.bot = bot

    @command()
    async def rag(self, ctx: Context):
        message = ctx.message
        message_without_command = message.content.replace("$rag", "")
        embed_msg = vectorize_msg(message_without_command)

        current_channel_settings = ctx.channel.topic
        txt_filter = None
        if current_channel_settings is not None and yaml.safe_load(current_channel_settings) is not None:
            tmp_yaml = yaml.safe_load(current_channel_settings)
            txt_allowed = tmp_yaml[yaml_key]
            txt_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="title",
                        match=models.MatchAny(
                            any=txt_allowed
                        ),
                    )
                ]
            )

        qdrant_search = client_qdrant.search(
            collection_name="contents_text",
            limit=50,
            query_vector=embed_msg,
            query_filter=txt_filter
        )
        scores = [x.score for x in qdrant_search]
        print(f"Before thresholding there is {len(scores)} hit")
        print(f"Score max: {max(scores)}, score min: {min(scores)}, score mean: {mean(scores)}")
        filtered_qs = [x for x in qdrant_search if x.score > 0.3]

        if len(filtered_qs) <= 0:
            await message.channel.send(f"Il n'y a pas assez de sources pour répondre. Score moyen trouvé : {mean(scores)}")

        sources = []
        for i, point in enumerate(filtered_qs):
            e = point.payload
            local_e = Embed(
                title=f"Source #{i+1} - {e['title']}",
                description=f"""
                            Score {point.score}
                            """,
                color=adler32(e['title'].encode("utf-8")) % 16777215
            )

            local_e.add_field(
                name="Approximative page number",
                value=f"{int(e['original_indice'])}"
            )
            local_e.add_field(
                name="Texte d'origine",
                value=e["orginal_text"][:200] + "..."
            )
            sources.append(local_e)

        clean_texts = [str({"clean_text": x.payload.get("orginal_text"), "title of book": x.payload.get("title")}) for
                       x
                       in filtered_qs]

        question_message = template_user_message.replace(
            "{retrieved_chunk}", "\n".join(clean_texts)
        ).replace(
            "{question}", message_without_command
        )

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
            if content is not None and len(content) > 0:
                accu_resp.append(content)
                if "\n" in content or len("".join(accu_resp)) >= 1500:
                    if accu_resp == "\n":
                        accu_resp = "----"
                    try:
                        await message.channel.send("".join(accu_resp))
                    except Exception as e:
                        print(str(e))
                    accu_resp.clear()
        await message.channel.send("".join(accu_resp))
        thread_msg = await message.channel.send("Thread pour les sources")

        source_thread = await message.channel.create_thread(
            name="Source Thread",
            message=thread_msg
        )

        for source in sources:
            await source_thread.send(embed=source)
