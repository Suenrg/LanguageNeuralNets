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
File = open('tokens.txt','r')
token = File.read()
File.close()

prefix = '?'

nn = STest()

goodPath = 'good.txt'
badPath = 'bad.txt'

client = discord.Client()

@client.event
async def on_ready():
    print('We have logged in as {0.user}'.format(client))

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content.startswith(prefix+'good'):
        phrase = message.content[6:]
        await client.send_message(message.channel, content = 'The phrase "' + phrase + '" is now in the good words list!')
        f = open(goodPath, "a")
        f.write(phrase + '\n')
        f.close()
        return
    if message.content.startswith(prefix+'bad'):
        phrase = message.content[5:]
        await client.send_message(message.channel, content = 'The phrase "' + phrase + '" is now in the bad words list!')
        f = open(badPath, "a")
        f.write(phrase + '\n')
        f.close()
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
