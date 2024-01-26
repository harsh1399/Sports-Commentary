import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import io, transforms
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor
from transformers import AutoTokenizer, GPT2Config, default_data_collator
from transformers import VivitImageProcessor, VivitModel
from huggingface_hub import hf_hub_download
import av
from pathlib import Path
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader, random_split

class config :
    ENCODER = "google/vivit-b-16x2-kinetics400"
    DECODER = "gpt2"
    TRAIN_BATCH_SIZE = 4
    VAL_BATCH_SIZE = 2
    VAL_EPOCHS = 1
    LR = 5e-5
    SEED = 42
    MAX_LEN = 128
    SUMMARY_LEN = 20
    WEIGHT_DECAY = 0.01
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)
    TRAIN_PCT = 0.95
    EPOCHS = 3
    IMG_SIZE = (224,224)
    LABEL_MASK = -100
    TOP_K = 1000
    TOP_P = 0.95

df = pd.read_csv('Data/commentary.csv')

def train_video_links(ball_number):
  return "Data/train/ball"+str(ball_number)+".mp4"

def test_video_links(ball_number):
  return "Data/test/ball"+str(ball_number)+".mp4"

def val_video_links(ball_number):
  return "Data/val/ball"+str(ball_number)+".mp4"

df['video'] = df[df['currentInning.id']==85915]['currentInning.balls'].iloc[:16].apply(train_video_links)
train_df = df[df['currentInning.id']==85915][['text','video']].iloc[:16]

df['video'] = df[df['currentInning.id']==85915]['currentInning.balls'].iloc[16:22].apply(test_video_links)
test_df = df[df['currentInning.id']==85915][['text','video']].iloc[16:22]

df['video'] = df[df['currentInning.id']==85915]['currentInning.balls'].iloc[22:26].apply(val_video_links)
val_df = df[df['currentInning.id']==85915][['text','video']].iloc[22:26]

np.random.seed(0)


def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    print(len(indices))
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    '''
    Sample a given number of frame indices from the video.
    Args:
        clip_len (`int`): Total number of frames to sample.
        frame_sample_rate (`int`): Sample every n-th frame.
        seg_len (`int`): Maximum allowed index of sample's last frame.
    Returns:
        indices (`List[int]`): List of sampled frame indices
    '''
    converted_len = int(clip_len * frame_sample_rate)
    if converted_len <= seg_len:
      end_idx = np.random.randint(converted_len, seg_len)
    else:
      end_idx = np.random.randint(seg_len,converted_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices

class ImgDataset(Dataset):
    def __init__(self, df, tokenizer, image_processor):
        self.df = df
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = 70

    def __len__(self, ):
        return len(self.df)

    def __getitem__(self, idx):
        print(self.df.video.iloc[idx])
        caption = self.df.text.iloc[idx]
        video_path = self.df.video.iloc[idx]
        # img_path = os.path.join(self.root_dir , image)
        # img = Image.open(img_path).convert("RGB")
        container = av.open(video_path)
        indices = sample_frame_indices(clip_len=32, frame_sample_rate=4, seg_len=container.streams.video[0].frames)
        video = read_video_pyav(container=container, indices=indices)

        # if self.transform is not None:
        #     img= self.transform(img)
        # pixel_values = self.feature_extractor(img, return_tensors="pt").pixel_values
        inputs = image_processor(list(video), return_tensors="pt").pixel_values
        captions = self.tokenizer(caption,
                                  padding='max_length', truncation=True).input_ids
        captions = [caption if caption != self.tokenizer.pad_token_id else -100 for caption in captions]
        print(inputs.size(), len(captions))
        encoding = {"pixel_values": inputs.squeeze(), "labels": torch.tensor(captions)}
        return encoding



tokenizer = AutoTokenizer.from_pretrained(config.DECODER)
image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")



train_dataset = ImgDataset(train_df, tokenizer,image_processor)
val_dataset = ImgDataset(val_df ,tokenizer,image_processor)
test_dataset = ImgDataset(test_df,tokenizer,image_processor)

video_model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(config.ENCODER,config.DECODER)

video_model.config.decoder_start_token_id = tokenizer.cls_token_id
video_model.config.pad_token_id = tokenizer.pad_token_id
# make sure vocab size is set correctly
video_model.config.vocab_size = video_model.config.decoder.vocab_size
# set beam search parameters
video_model.config.eos_token_id = tokenizer.sep_token_id
video_model.config.decoder_start_token_id = tokenizer.bos_token_id
video_model.config.max_length = 128
video_model.config.early_stopping = True
video_model.config.no_repeat_ngram_size = 3
video_model.config.length_penalty = 2.0
video_model.config.num_beams = 4


import datasets
rouge = datasets.load_metric("rouge")

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    # all unnecessary tokens are removed
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid

    return {
        "rouge2_precision": round(rouge_output.precision, 4),
        "rouge2_recall": round(rouge_output.recall, 4),
        "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
    }

training_args = Seq2SeqTrainingArguments(
    output_dir='VIT_large_gpt2',
    predict_with_generate=True,
    evaluation_strategy="epoch",
    do_train=True,
    do_eval=True,
    logging_steps=1024,
    save_steps=2048,
    warmup_steps=1024,
    learning_rate = 5e-5,
    #max_steps=1500, # delete for full training
    num_train_epochs = config.EPOCHS, #TRAIN_EPOCHS
    overwrite_output_dir=True,
    save_total_limit=1,
)

tokenizer.pad_token = tokenizer.unk_token
trainer = Seq2SeqTrainer(
    tokenizer=image_processor,
    model=video_model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=default_data_collator,
)
trainer.train()