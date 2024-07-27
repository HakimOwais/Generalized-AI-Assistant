import os
import sys
import telegram
import asyncio
from dotenv import load_dotenv
from source.summarization import *

load_dotenv()

load_dotenv()

sys.path.append("source")
sys.path.append("application")
sys.path.append("configuration")


my_token = os.getenv("TELEGRAM_TOKEN")
my_chat_id = 5654807603

returned_data = summarize("123asd")

async def send(msg, chat_id, token=my_token):


    bot = telegram.Bot(token=token)
    await bot.sendMessage(chat_id=chat_id, text=msg)
    print("Message Sent!")


MessageString = "Hello, People!"
print(MessageString)
asyncio.run(send(msg=returned_data, chat_id=my_chat_id, token=my_token))



