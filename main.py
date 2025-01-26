# Telegram AI Bot using Python and Tensorflow
from telegram.ext import Application, CommandHandler, MessageHandler, filters
from io import BytesIO
import cv2
import numpy as np 
import tensorflow as tf 
import threading

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


def start(update, context):
    update.message.reply_text("Welcome to the Telegram Bot!")

def help(update, context):
    update.message.reply_text("""
    /start - Starts the conversation
    /help - Shows this message
    /train - trains the neural network

    """)

def train(update, context):
    update.message.reply_text("Model is being trained...")

    def train_model():
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        model.fit(x_train, y_train, epochs=10, validation_data=[x_test, y_test])
        model.save("cifar_classifier.model")
        update.message.reply_text("Done! You can now send a photo if you wish.")

    threading.Thread(target=train_model).start()
    

def handle_message(update, context):
    update.message.reply_text("Please train the model and send a picture.")

def handle_photo(update, context):
    try:
        file = context.bot.get_file(update.message.photo[-1].file_id)
        file_bytes = np.frombuffer(file.download_as_bytearray(), dtype=np.uint8)

        img  = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)

        prediction = model.predict(np.array([img / 255]))
        update.message.reply_text(f"In this image I see a {class_names[np.argmax(prediction)]}")
    except Exception as e:
        update.message.reply_text(f"Error processing this image: {e}")




app = Application.builder().token(TOKEN).build()

app.add_handler(CommandHandler("start",  start))
app.add_handler(CommandHandler("help",  help))
app.add_handler(CommandHandler("train",  train))
app.add_handler(MessageHandler(filters.TEXT, handle_message))
app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

app.run_polling()