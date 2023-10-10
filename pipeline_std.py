# import everything
from PIL import Image
import easyocr

from text_to_speech.runner import get_speech

# img path
img_fn = "C:/Users/Nadia Timoleon/Documents/GitHub/ImageText-to-Speech/scene_text_test_img.jpg"

# load image
image = Image.open(img_fn)

# loading model
reader = easyocr.Reader(['en'], gpu=False)

# scene text detection from image
result = reader.readtext(img_fn)
generated_text = ' '.join([result[i][1] for i in range(len(result))])
print("detected text:", generated_text)

# text to speech processing
get_speech(generated_text, "handwritten_audio.mp3")
