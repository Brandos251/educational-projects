from tfidf import get_answer
from parsing import parse
import telebot
import schedule
import time
import threading
bot = telebot.TeleBot('7959196680:AAEucQz4_ehZIfHEGJadbA-bz8A_d3pOd4g')

schedule.every(10).minutes.do(parse)

def start_parse():
    while True:
        schedule.run_pending()
        time.sleep(1)

t1 = threading.Thread(target=start_parse, args=(), daemon=True)
t1.start()

@bot.message_handler(content_types=['text'])
def get_text_messages(message):
    if message.text == "/help":
        bot.send_message(message.from_user.id, "Напиши вопрос, который тебя интерисует")
    else:
        bot.send_message(message.from_user.id, get_answer(message.text))

bot.polling(none_stop=True, interval=0)