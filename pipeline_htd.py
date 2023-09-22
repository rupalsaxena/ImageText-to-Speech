# import everything
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from text_to_speech.runner import get_speech

# img path, model path
img_fn = "/Users/rupalsaxena/Documents/GitHub/ImageText-to-Speech/handwritten_test_img.png"
model_fn = "/Users/rupalsaxena/Documents/GitHub/ImageText-to-Speech/trocr_finetuned_model"

# load image
image = Image.open(img_fn).convert("RGB")

# preprocess image using pretrained model (if required)
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
pixel_values = processor(image, return_tensors="pt").pixel_values

# loading finetuned model
model = VisionEncoderDecoderModel.from_pretrained(model_fn)

# handwritten text detection from image
generated_ids = model.generate(pixel_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("detected text:", generated_text)

# text to speech processing
get_speech(generated_text, "handwritten_audio.mp3")
