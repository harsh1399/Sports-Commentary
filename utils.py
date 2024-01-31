import numpy as np
import pandas as pd
import av
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import VivitImageProcessor, VivitModel
from transformers import AutoTokenizer, GPT2Config, default_data_collator
from config import config
import datasets
import torch
from sklearn.model_selection import train_test_split
import os

np.random.seed(25)
rouge = datasets.load_metric("rouge")
tokenizer = AutoTokenizer.from_pretrained(config.DECODER)
tokenizer.pad_token = tokenizer.unk_token
image_processor = VivitImageProcessor.from_pretrained(config.ENCODER)


def get_tokenizer():
    return AutoTokenizer.from_pretrained(config.DECODER)


def get_image_processor():
    return VivitImageProcessor.from_pretrained(config.ENCODER)

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
    # print(len(indices))
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
    start_idx = 0
    if converted_len <= seg_len:
        end_idx = np.random.randint(converted_len, seg_len)
        start_idx = end_idx - converted_len
    else:
        end_idx = seg_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices


def train_video_links(ball_number):
  return "Data/train/ball"+str(ball_number)+".mp4"


def test_video_links(ball_number):
  return "Data/test/ball"+str(ball_number)+".mp4"


def val_video_links(ball_number):
  return "Data/val/ball"+str(ball_number)+".mp4"


class ImgDataset(Dataset):
    def __init__(self, df, tokenizer, image_processor,trainortest,outputdir):
        self.df = df
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = config.MAX_LEN
        self.data_use = trainortest
        self.output_dir = outputdir
    def __len__(self, ):
        return len(self.df)

    def __getitem__(self, idx):
        if self.data_use == "test":
            with open(f"{self.output_dir}/commentary_final.txt", 'a') as f:
                f.write(self.df.file.iloc[idx]+ "\n")
        caption = self.df.commentary.iloc[idx]
        video_path = self.df.file.iloc[idx]
        # img_path = os.path.join(self.root_dir , image)
        # img = Image.open(img_path).convert("RGB")
        container = av.open(video_path)
        indices = sample_frame_indices(clip_len=32, frame_sample_rate=4, seg_len=container.streams.video[0].frames)
        video = read_video_pyav(container=container, indices=indices)

        # if self.transform is not None:
        #     img= self.transform(img)
        # pixel_values = self.feature_extractor(img, return_tensors="pt").pixel_values
        inputs = self.image_processor(list(video), return_tensors="pt").pixel_values
        captions = self.tokenizer(caption,padding='max_length',max_length = self.max_length, truncation=True).input_ids
        captions = [caption if caption != self.tokenizer.pad_token_id else -100 for caption in captions]
        # print(inputs.size(), len(captions))
        encoding = {"pixel_values": inputs.squeeze(), "labels": torch.tensor(captions)}
        return encoding


def combine_video_and_commentary(df,video_files):
    new_df = {'commentary':[],'file':[]}
    for file in video_files:
        file_name = file.split('.')[0]
        ball_number = int(file_name[4:])
        # innings_data = df[df['currentInning.id'] == 85915]
        ball_data = df[df['currentInning.balls'] == ball_number]
        new_df['commentary'].append(ball_data.iloc[0]['text'])
        new_df['file'].append(f'Data/videos/{file}')
    return pd.DataFrame(new_df)

def get_dataset(output_dir):
    df = pd.read_csv('Data/updated_commentary.csv')
    video_files = os.listdir("Data/videos")
    new_df = combine_video_and_commentary(df,video_files)

    train_df,test_df = train_test_split(new_df,test_size=0.2)
    train_df,val_df = train_test_split(train_df,test_size=0.15)
    train_dataset = ImgDataset(train_df, tokenizer, image_processor,"train",output_dir)
    val_dataset = ImgDataset(val_df, tokenizer, image_processor,"val",output_dir)
    test_dataset = ImgDataset(test_df, tokenizer, image_processor,"test",output_dir)
    return train_dataset,val_dataset,test_dataset


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

# get_dataset()