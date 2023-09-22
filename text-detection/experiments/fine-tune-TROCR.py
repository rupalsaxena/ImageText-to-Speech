import pandas as pd
import torch
from evaluate import load
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW, TrOCRProcessor, VisionEncoderDecoderModel

from helpers import compute_cer
from IAMDataset import IAMDataset

df = pd.read_csv("/Users/rupalsaxena/Documents/GitHub/ImageText-to-Speech/text-detection/experiments/data/IAM/gt.csv")

# split the data
train_df, test_df = train_test_split(df, test_size=0.2)

# we reset the indices to start from zero
train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)

# processer
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")

# dataset creation
train_dataset = IAMDataset(root_dir='/Users/rupalsaxena/Documents/GitHub/ImageText-to-Speech/text-detection/experiments/data/IAM/self_lines',
                           df=train_df,
                           processor=processor)
eval_dataset = IAMDataset(root_dir='/Users/rupalsaxena/Documents/GitHub/ImageText-to-Speech/text-detection/experiments/data/IAM/self_lines',
                           df=test_df,
                           processor=processor)

print("Number of training examples:", len(train_dataset))
print("Number of validation examples:", len(eval_dataset))

encoding = train_dataset[0]
for k,v in encoding.items():
  print(k, v.shape)

# create dataoader
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=4)

#model init
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")
model.to(device)

# set special tokens used for creating the decoder_input_ids from the labels
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
# make sure vocab size is set correctly
model.config.vocab_size = model.config.decoder.vocab_size

# set beam search parameters
model.config.eos_token_id = processor.tokenizer.sep_token_id
model.config.max_length = 64
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.length_penalty = 2.0
model.config.num_beams = 4

# load metrics
cer_metric = load("cer")

# train model
optimizer = AdamW(model.parameters(), lr=5e-5)

for epoch in range(2):  # loop over the dataset multiple times
    # train
    model.train()
    train_loss = 0.0
    for batch in tqdm(train_dataloader):
        # get the inputs
        for k,v in batch.items():
            batch[k] = v.to(device)

        # forward + backward + optimize
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()

    print(f"Loss after epoch {epoch}:", train_loss/len(train_dataloader))
        
    # evaluate
    model.eval()
    valid_cer = 0.0
    with torch.no_grad():
        for batch in tqdm(eval_dataloader):
            # run batch generation
            outputs = model.generate(batch["pixel_values"].to(device))
            # compute metrics
            cer = compute_cer(cer_metric, processor, pred_ids=outputs, label_ids=batch["labels"])
            valid_cer += cer 

    print("Validation CER:", valid_cer / len(eval_dataloader))

model.save_pretrained(".")
