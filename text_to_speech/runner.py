import os

from gtts import gTTS


def run_gTTS(text, language, output_fn, filename): 
    speech = gTTS(text=text, lang=language, slow=False, tld="com.au")
    if not os.path.exists(output_fn):
        os.makedirs(output_fn)
    filename = os.path.join(output_fn, filename)
    speech.save(filename)

def get_speech(text, filename):
    language = "en"
    output_fn = "audio-output"
    run_gTTS(text=text, language=language, output_fn=output_fn, filename=filename)


# example run
filename = "gtts_speech.mp3"
text = "hello! how is it going"
get_speech(text, filename)
