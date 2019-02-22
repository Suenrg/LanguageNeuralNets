#Discord Bot
#Ben Shakow
#with help from https://discordpy.readthedocs.io

import discord
from discord.ext import commands

#Receive token from hidden file
File = open('C:/Code/LanguageNeuralNets/tokens.txt','r')
token = File.read()
File.close()

bot = commands.Bot(command_prefix='$')

client = discord.Client()

@client.event
async def on_ready():
    print('We have logged in as {0.user}'.format(client))

@bot.command()
async def test(ctx, arg):
    await ctx.send('The worde you sent was ' + arg)


client.run(token.strip())
