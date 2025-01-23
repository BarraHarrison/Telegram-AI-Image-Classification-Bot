# Telegram AI Bot using Python and Tensorflow
from telegram.ext import *
from io import BytesIO
import cv2
import numpy as np 
import tensorflow as tf 

with open("token.txt", "r") as f:
    TOKEN = f.read()

def start(update, context):
    update.message.reply_text("Welcome to the Telegram Bot!")

def help(update, context):
    update.message.reply_text("""
    /start - Starts the conversation
    /help - Shows this message
    /train - trains the neural network

    """)

def train(update, context):
    pass

def handle_message(update, context):
    update.message.reply_text("Please train the model and send a picture.")

def handle_photo(update, context):
    pass

updater = Updater(TOKEN, use_context=True)
dp = updater.dispatcher

dp.add_handler(CommandHandler("start",  start))
dp.add_handler(CommandHandler("help",  help))
dp.add_handler(CommandHandler("train",  train))
dp.add_handler(MessageHandler(Filters.text, handle_message))