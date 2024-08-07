import os
from zlib import adler32

import yaml
from discord import Embed
from discord.ext.commands import Context, Cog, command
from dotenv import load_dotenv
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from sentence_transformers import SentenceTransformer
from qdrant_client.http.models import models

from helpers.qdrant_helper import setup_client_qdrant

load_dotenv()  # take environment variables from .env.

system_message = open("prompt/client_discord/system.txt", "r", encoding="utf-8").read()
template_user_message = open("prompt/client_discord/user.txt", "r", encoding="utf-8").read()

MODEL = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

client_qdrant = setup_client_qdrant()

mistral_api_key = os.environ["MISTRAL_API_KEY"]
model = "open-mixtral-8x22b"

client_mistral = MistralClient(api_key=mistral_api_key)
yaml_key = "Textes actifs"


class HandleSearch(Cog):
    def __init__(self, bot):
        self.bot = bot

    @command()
    async def rag(self, ctx: Context):
        message = ctx.message
        message_without_command = message.content.replace("$rag", "")
        embed_msg = MODEL.encode(message_without_command, show_progress_bar=True)

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

            local_e.add_field(
                name="Approximative page number",
                value=f"{int(e["page_index"])}"
            )
            local_e.add_field(
                name="Texte d'origine",
                value=e["original_content"][:200] + "..."
            )
            sources.append(local_e)

        clean_texts = [str({"clean_text": x.payload.get("original_content"), "title of book": x.payload.get("title")}) for
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
