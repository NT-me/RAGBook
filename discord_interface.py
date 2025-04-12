# This example requires the 'message_content' intent.
import os

import discord
from dotenv import load_dotenv
from modules.handle_search import HandleSearch
from modules.handle_texts import HandleText
from discord.ext import commands

load_dotenv()  # take environment variables from .env.

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix='$', intents=intents)


@bot.event
async def on_ready():
    await bot.add_cog(HandleSearch(bot))
    await bot.add_cog(HandleText(bot))
    print(f'We have logged in as {bot.user}')


if __name__ == "__main__":
    bot.run(os.environ["DISCORD_API_KEY"])
