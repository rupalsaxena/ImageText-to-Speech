# import everything
from PIL import Image
import easyocr

from text_to_speech.runner import get_speech

# img path
img_fn = "C:/Users/Nadia Timoleon/Documents/GitHub/ImageText-to-Speech/scene_text_test_img.jpg"

# load image
image = Image.open(img_fn)

# scene text detection from image
reader = easyocr.Reader(['en'], gpu=False)
result = reader.readtext(img_fn)

# sorting text from top to bottom based on bounding box upper left point
text = [result[i][1] for i in range(len(result))]
upper_left_point = [result[i][0][0] for i in range(len(result))]
sorted_text = [x for x, _ in sorted(zip(text, upper_left_point), key=lambda x: x[1][1])]

generated_text = ' '.join([t for t in sorted_text if t != ''])
print("detected text:", generated_text)

# text to speech processing
get_speech(generated_text, "scene_text_audio.mp3")
