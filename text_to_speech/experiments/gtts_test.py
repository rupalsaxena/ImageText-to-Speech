import os
from gtts import gTTS

language = "en"
text = "Hello! How are you?"

speech = gTTS(text=text, lang=language, slow=False, tld="com.au")
foldername = "audio-output"
if not os.path.exists(foldername):
   os.makedirs(foldername)
filename = os.path.join(foldername, "gtts_speech.mp3")
speech.save(filename)
