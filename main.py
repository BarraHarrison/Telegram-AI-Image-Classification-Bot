# Telegram AI Bot using Python and Tensorflow
from telegram.ext import Application, CommandHandler, MessageHandler, filters
from io import BytesIO
import cv2
import numpy as np 
import tensorflow as tf 
import threading
import logging

logging.basicConfig(level=logging.INFO)

with open("token.txt", "r") as f:
    TOKEN = f.read()


# Convolutional Neural Network
# Loading the training and testing data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train.astype("float32") / 255.0, x_test.astype("float32") / 255.0

class_names = ["Plane", "Car", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=(32, 32, 3)))
model.add(tf.keras.layers.MaxPooling2D((2,2)))
model.add(tf.keras.layers.Conv2D(64, (3,3), activation="relu"))
model.add(tf.keras.layers.MaxPooling2D((2,2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation="relu"))
model.add(tf.keras.layers.Dense(10, activation="softmax"))


async def start(update, context):
    await update.message.reply_text("Welcome to the Telegram Bot!")

async def help(update, context):
    await update.message.reply_text("""
    /start - Starts the conversation
    /help - Shows this message
    /train - trains the neural network

    """)

async def stop(update, context):
    await update.message.reply_text("Stopping the bot. Goodbye and have a nice day!")
    await context.application.stop()


async def train(update, context):
    await update.message.reply_text("Model is being trained...")

    def train_model():
        try:
            model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            model.fit(x_train, y_train, epochs=10, validation_data=[x_test, y_test])
            model.save("cifar_classifier.keras")
        except Exception as e:
            print(f"Error during training: {e}")

    threading.Thread(target=train_model).start()
    

async def handle_message(update, context):
    await update.message.reply_text("Please train the model and send a picture.")

async def handle_photo(update, context):
    try:
        file = await context.bot.get_file(update.message.photo[-1].file_id)
        file_bytes = await file.download_as_bytearray()
        file_bytes = np.frombuffer(file_bytes, dtype=np.uint8)

        img  = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)

        prediction = model.predict(np.array([img / 255.0]))
        predicted_class = class_names[np.argmax(prediction)]
        await update.message.reply_text(f"In this image I see a {predicted_class}")
    except Exception as e:
        await update.message.reply_text(f"Error processing this image: {e}")




app = Application.builder().token(TOKEN).build()

app.add_handler(CommandHandler("start",  start))
app.add_handler(CommandHandler("help",  help))
app.add_handler(CommandHandler("train",  train))
app.add_handler(CommandHandler("stop", stop))
app.add_handler(MessageHandler(filters.TEXT, handle_message))
app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

app.run_polling()