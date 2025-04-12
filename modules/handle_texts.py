from discord.ext.commands import Context, Cog, command
from dotenv import load_dotenv
from qdrant_client.http.models import models

from helpers.qdrant_helper import setup_client_qdrant
import yaml

load_dotenv()  # take environment variables from .env.

client_qdrant = setup_client_qdrant()

yaml_key = "Textes actifs"


async def get_texts_names():
    set_oftexts_name: set = set()
    while True:
        q_ret = client_qdrant.scroll(
            collection_name="contents_text",
            scroll_filter=models.Filter(
                must_not=[
                    models.FieldCondition(
                        key="title",
                        match=models.MatchAny(any=list(set_oftexts_name)),
                    ),
                ]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        points = q_ret[0]

        if len(points) == 0:
            break

        set_oftexts_name.add(points[0].payload.get("title"))
    return set_oftexts_name


class HandleText(Cog):
    def __init__(self, bot):
        self.bot = bot

    @command()
    async def list_texts(self, ctx: Context):
        set_oftexts_name = await get_texts_names()

        await ctx.channel.send("Les textes dipsonibles sont :\n" + "\n".join([f"- *{x}*" for x in set_oftexts_name]))

    @command()
    async def add_text(self, ctx: Context, text_name: str):
        txt_names = await get_texts_names()

        if text_name not in txt_names:
            await ctx.channel.send("Error in the name, it doesnt exist in the database")

        current_channel_settings = ctx.channel.topic

        if current_channel_settings is None or yaml.safe_load(current_channel_settings) is None:
            new_channel_settings = f"{yaml_key}:\n- {text_name}\n"
            await ctx.channel.edit(reason="New text", topic=new_channel_settings)

        else:
            tmp_yaml = yaml.safe_load(current_channel_settings)
            if text_name not in tmp_yaml[yaml_key]:
                tmp_yaml[yaml_key].append(text_name)
                new_channel_settings = yaml.dump(tmp_yaml, indent=4, default_flow_style=False)
                await ctx.channel.edit(reason="New text", topic=new_channel_settings)

        await ctx.channel.send("New text added")

    @command()
    async def delete_text(self, ctx: Context, text_name: str):
        txt_names = await get_texts_names()

        if text_name not in txt_names:
            await ctx.channel.send("Error in the name, it doesnt exist in the database")

        current_channel_settings = ctx.channel.topic

        if current_channel_settings is None or yaml.safe_load(current_channel_settings) is None:
            await ctx.channel.send("Nothing to delete")

        else:
            tmp_yaml = yaml.safe_load(current_channel_settings)
            if text_name in tmp_yaml[yaml_key]:
                tmp_yaml[yaml_key].remove(text_name)
                new_channel_settings = yaml.dump(tmp_yaml, indent=4, default_flow_style=False)
                await ctx.channel.edit(reason="New text", topic=new_channel_settings)
            else:
                await ctx.channel.send("Nothing to delete")
