import torch
from config import config
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import VisionEncoderDecoderModel
from transformers import default_data_collator
import utils
import argparse

model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(config.ENCODER,config.DECODER)

model.config.decoder_start_token_id = utils.tokenizer.cls_token_id
model.config.pad_token_id = utils.tokenizer.pad_token_id
# make sure vocab size is set correctly
model.config.vocab_size = model.config.decoder.vocab_size
# set beam search parameters
model.config.eos_token_id = utils.tokenizer.sep_token_id
model.config.decoder_start_token_id = utils.tokenizer.bos_token_id
model.config.max_length = 128
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.length_penalty = 2.0
model.config.num_beams = 4

def train_model(output_dir):
    training_args = Seq2SeqTrainingArguments(
        output_dir=f"{output_dir}/VIT_large_gpt2",
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
    train_dataset,val_dataset,test_dataset = utils.get_dataset()

    trainer = Seq2SeqTrainer(
        tokenizer=utils.image_processor,
        model=model,
        args=training_args,
        compute_metrics=utils.compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
    )
    trainer.train()
    trainer.save_model(f'{output_dir}/VIT_large_gpt2')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, help='Directory where model checkpoints will be saved')
    args = parser.parse_args()
    output_dir = args.output_dir
    if torch.cuda.is_available():

        device = torch.device("cuda")

        print('There are %d GPU(s) available.' % torch.cuda.device_count())

        print('We will use the GPU:', torch.cuda.get_device_name(0))

    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    train_model(output_dir)