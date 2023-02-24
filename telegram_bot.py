import telegram_bot_token
import detect
from ultralytics import YOLO
from keras.models import load_model
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
from telegram import Update
import cv2
import logging
import os
from PIL import Image


def main():
    async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "Send me ONE image and I will tell if there are fractures in the glass! Pink colour is where there are cracks.")

    async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "My only purpose is to tell you if there are fractures in the glass, so please send ONE image! Pink colour is where there are cracks.")

    async def detect_cracks(update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("Image received... Please wait a while!")

        # Get the image
        # user = update.message.from_user
        # photo_file = await update.message.photo[-1].get_file()
        # await photo_file.download('user_photo.jpg')
        photo_file = await update.message.photo[-1].get_file()
        file_name = photo_file.file_path if photo_file.file_path is not None else photo_file.file_id
        name = file_name.split("/")[-1:][0]
        extension = name.split(".")[-1:][0].lower()
        file = await photo_file.download_to_drive()

        if extension != "jpg":
            await update.message.reply_text("I cannot accept it, please send in .jpg format!")
        else:
            # Convert image into array form
            image = cv2.imread(name)

            # Use the detect function to
            crack = detect.detect(image, model, [
                trained_vgg16_crack, trained_resnet101_crack, trained_inceptionv3_crack])

            if crack == "No glass!":
                await update.message.reply_text("There is no glass!")
            else:
                crack = crack[:, :, ::-1]
                Image.fromarray(crack).save('results.jpg')
                await context.bot.send_document(chat_id=update.message['chat']['id'], document='./results.jpg')
        os.remove(name)

    # Get Token and initiate the bot
    print("Initialising everything...")
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
    logger = logging.getLogger(__name__)
    token = telegram_bot_token.getToken()
    application = Application.builder().token(token).build()
    # updater = Updater()  # Token API from Telegram
    # dispatcher = updater.dispatcher

    # Load every model
    best = 'best.pt'
    model = YOLO(model=best)
    trained_vgg16_crack = load_model('./VGG16_Crack_BW')
    trained_resnet101_crack = load_model('./ResNet101_Crack_BW')
    trained_inceptionv3_crack = load_model('./InceptionV3_Crack_BW')

    # Handlers
    print("Adding handlers...")
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(
        filters.PHOTO, detect_cracks))

    # Stops the bot if Ctrl + C is pressed
    print("Done!")
    application.run_polling()


if __name__ == "__main__":
    main()
# For receiving files (i.e. pictures)
# telegram_link = 'https://api.telegram.org/file/bot' + token + "/"  # + file_path
