import sys
import telegram
import asyncio
from source.summarization import *


sys.path.append("source")
sys.path.append("application")
sys.path.append("configuration")


my_token = "7017880287:AAE6qEx1qbNDwgKdglJR5nZKW-oL7hmPXYw"
my_chat_id = 5654807603

returned_data = summarize("123")

async def send(msg, chat_id, token=my_token):


    bot = telegram.Bot(token=token)
    await bot.sendMessage(chat_id=chat_id, text=msg)
    print("Message Sent!")


MessageString = "Hello, People!"
print(MessageString)
asyncio.run(send(msg=returned_data, chat_id=my_chat_id, token=my_token))



