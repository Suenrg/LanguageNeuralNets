#Discord Bot
#Ben Shakow
#with help from https://discordpy.readthedocs.io
# from __future__ import absolute_import, division, print_function
import discord
from discord.ext import commands
# import matplotlib.pyplot as plt
# import random
# from keras.models import load_model
# import tensorflow as tf
# import keras
# from keras.preprocessing.text import Tokenizer
# import numpy as np
# import pickle
from STest import STest

#Receive token from hidden file
File = open('C:/Code/LanguageNeuralNets/tokens.txt','r')
token = File.read()
File.close()

prefix = '?'

nn = STest()
print(nn.token)

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
        return
    if message.content.startswith(prefix+'hewwo'):
        await client.send_message(message.channel, content = "Hey pwease stop! owo")
        return
    if message.content.startswith(prefix+'p'):
        phrase = message.content[2:]
        result = nn.testPhrase(phrase)
        msg = 'The phrase "'+ phrase + '" is '
        if(result[0]<=.5):
            msg += 'bad :('
        else:
            msg += 'good!'
        await client.send_message(message.channel, content = msg)
        return

client.run(token.strip())
