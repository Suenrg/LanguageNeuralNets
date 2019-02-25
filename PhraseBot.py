#Discord Bot
#Ben Shakow
#with help from https://discordpy.readthedocs.io
import discord
from discord.ext import commands
from STest import STest

#Receive token from hidden file
File = open('tokens.txt','r')
token = File.read()
File.close()


#Bring in our neural network
nn = STest()

#set up file paths / prefixes
prefix = '?'
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
    #these two are for adding to the bad/good lists
    if message.content.startswith(prefix+'good'):
        phrase = message.content[6:]
        await client.send_message(message.channel, content = 'The phrase "' + phrase + '" is now in the good words list!')
        f = open(goodPath, "a")
        f.write(phrase + '\n')
        f.close()
        return
    if message.content.startswith(prefix+'recalibrate'):
        await client.send_message(message.channel, content = 'Recalibrating the neural network')
        exec(open("sentiment_clean.py").read())
        File = open('acc','r')
        acc = File.read()
        File.close()
        await client.send_message(message.channel, content = 'Neural Network recalibrated, with an accuracy of '+acc)
        return
    if message.content.startswith(prefix+'bad'):
        phrase = message.content[5:]
        await client.send_message(message.channel, content = 'The phrase "' + phrase + '" is now in the bad words list!')
        f = open(badPath, "a")
        f.write(phrase + '\n')
        f.close()
        return
    #for actually checking a phrase
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
