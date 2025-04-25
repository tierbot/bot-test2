import discord
import random
from discord.ext import commands

intents = discord.Intents.default()
intents.message_content = True
intents.members = True


from keras.models import load_model
from PIL import Image, ImageOps 
from io import BytesIO 
import numpy as np


np.set_printoptions(suppress=True)
model = load_model("keras_model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)\

bot = commands.Bot(command_prefix='$', intents=intents)

def preprocess_image(image):
    image = image.resize((224, 224)) 
    image = np.array(image).astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@bot.event
async def on_ready():
    print(f'We have logged in as {bot.user}')

@bot.command(name='identify')
async def identify(ctx):
    if ctx.message.attachments:
        for attachment in ctx.message.attachments:
            if attachment.filename.lower().endswith(('jpg', 'jpeg', 'png')):
                img_bytes = await attachment.read()
                img = Image.open(BytesIO(img_bytes)).convert('RGB')

                processed = preprocess_image(img)
                prediction = model.predict(processed)
                pred_index = np.argmax(prediction)
                pred_class = class_names[pred_index]
                confidence = float(prediction[0][pred_index])

                await ctx.send(
                    f"🕊️ I'm {confidence * 100:.2f}% sure that's a **{pred_class}**!"
                )
                return

@bot.command()
async def hello(ctx):
    await ctx.send(f'Hi! I am a bot {ctx.author.mention}!')

@bot.command()
async def heh(ctx, count_heh = 5):
    await ctx.send("he" * count_heh)

@bot.command()
async def joined(ctx, member: discord.Member):
    """Says when a member joined."""
    await ctx.send(f'{member.name} joined {discord.utils.format_dt(member.joined_at)}')

@bot.command()
async def add(ctx, left: int, right: int):
    """Adds two numbers together."""
    await ctx.send(left + right)

bot.run("TOKEN")