#Discord Bot
#Ben Shakow
#with help from https://discordpy.readthedocs.io

import discord
from discord.ext import commands

#Receive token from hidden file
File = open('C:/Code/LanguageNeuralNets/tokens.txt','r')
token = File.read()
File.close()

prefix = '!'

client = discord.Client()

@client.event
async def on_ready():
    print('We have logged in as {0.user}'.format(client))

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content.startswith(prefix+'hello'):
        await client.send_message(message.channel, content = "Hello!")
    if message.content.startswith(prefix+'hewwo'):
        await client.send_message(message.channel, content = "Hey pwease stop! owo")
    if message.content.startswith(prefix+'p'):
        
        await client.send_message(message.channel, content = "Hello!")

client.run(token.strip())
