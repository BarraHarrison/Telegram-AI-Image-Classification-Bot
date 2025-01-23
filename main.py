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