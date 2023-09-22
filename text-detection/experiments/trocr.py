from PIL import Image
from transformers import (GPT2TokenizerFast, TrOCRProcessor,
                          VisionEncoderDecoderModel, ViTImageProcessor)

# load image
img_fn = "/Users/rupalsaxena/Documents/GitHub/ImageText-to-Speech/text-detection/experiments/data/text-img.jpg"
image = Image.open(img_fn) 

def trocr_handwritten(image):
    #preprocess data
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    # calling the processor is equivalent to calling the feature extractor
    pixel_values = processor(image, return_tensors="pt").pixel_values
    print(pixel_values.shape)


    # load model
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

    # generate text
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(generated_text)
