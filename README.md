# ImageText-to-Speech
This project aims to support individuals with varying degrees of visual impairment, enabling them to access textual information contained within images. Shown below is the pipeline of this project:


![img](https://github.com/rupalsaxena/ImageText-to-Speech/blob/main/plots/workflow.png)

For each mode, we have developed a jupyter notebook to provide a more interactive overview of our workflow.

## Scene-Text Detection
For Scene-Text Detection, we have used [TextOCR dataset](https://textvqa.org/textocr/). Please download the data from [here](https://uzh-my.sharepoint.com/personal/konstantina_timoleon_uzh_ch/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fkonstantina%5Ftimoleon%5Fuzh%5Fch%2FDocuments%2F3rd%20Semester%2FEssentials%20in%20Text%20and%20Speech%20Processing%2Fscene%5Ftext%5Fdetection%5Fdata&ga=1).
We compared three OCR tools:

1. [pytesseract](https://pypi.org/project/pytesseract/)
2. [EasyOCR](https://www.jaided.ai/easyocr/tutorial/)
3. [keras-ocr](https://keras-ocr.readthedocs.io/en/latest/)

The data folder also includes our results on text detection and method comparison in pickle format which can be loaded for inspection in the Scene-Text Detection notebook.

## Handwritten-Text Detection
For Handwritten-Text Detection, we used [IAM Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database#icdar02). Please download the data from [here](https://drive.google.com/drive/folders/1wyugoG02loRRZxBy1lSlxGP9iUIylUof). For training, we fine-tuned HuggingFace [TrOCR base handwritten model](https://arxiv.org/abs/2109.10282) with IAM Handwriting Database. 

Trained model can be downloaded from [here](https://drive.google.com/drive/folders/1zyJJtwI9xbyJJVeXs3k7xxDTXL08XnGI). 

## Text-to-Speech Conversion
For Text-to-Speech Conversion, we used Google Text-to-Speech ([gTTS](https://gtts.readthedocs.io/en/latest/)) to tranform identified text into speech.

## Pipelines
We provide two Python scripts, with the complete, end-to-end pipeline for each text-detection pipeline.

1. Scene-Text-Images to Speech Conversion: Testing pipeline can be found here [here](https://github.com/rupalsaxena/ImageText-to-Speech/blob/main/pipeline_std.py)

2. Handwritten-Text-Images to Speech Conversion: Testing pipeline can be found here [here](https://github.com/rupalsaxena/ImageText-to-Speech/blob/main/pipeline_htd.py)
