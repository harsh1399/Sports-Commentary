import torch
from config import config
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import VisionEncoderDecoderModel,VivitConfig, VisionEncoderDecoderConfig, AutoConfig,GPT2Config
from transformers import default_data_collator
import utils
import argparse

device = None
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

vivit_config = VivitConfig()
vivit_config.output_hidden_states = True
vivit_config.return_dict = False
vivit_config.output_attentions = True
# roberta_config = AutoConfig.from_pretrained(config.DECODER)
# roberta_config.is_decoder = True
gpt2_config = GPT2Config()
encoder_decoder_config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(vivit_config,gpt2_config)

model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(config.ENCODER,config.DECODER, config = encoder_decoder_config)

model.config.decoder_start_token_id = utils.tokenizer.cls_token_id
model.config.pad_token_id = utils.tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size
model.config.eos_token_id = utils.tokenizer.sep_token_id
model.config.decoder_start_token_id = utils.tokenizer.bos_token_id
model.config.max_length = config.MAX_LEN
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.length_penalty = 2.0
model.config.num_beams = 4
model.config.output_hidden_states = False

def train_model(output_dir):
    training_args = Seq2SeqTrainingArguments(
        output_dir=f"{output_dir}/VIT_large_gpt2",
        predict_with_generate=True,
        evaluation_strategy="epoch",
        per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=config.VAL_BATCH_SIZE,
        # gradient_accumulation_steps = 4,
        do_train=True,
        do_eval=True,
        logging_steps=1024,
        save_steps=2048,
        warmup_steps=1024,
        learning_rate = config.LR,
        num_train_epochs = config.EPOCHS, #TRAIN_EPOCHS
        overwrite_output_dir=True,
        save_total_limit=1,
    )
    train_dataset,val_dataset,test_dataset = utils.get_dataset(output_dir)

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
    inference(test_dataset,output_dir)

def inference(test_dataset,output_dir):
    total_rouge_score = 0
    no_of_test_samples = len(test_dataset)
    for idx in range(no_of_test_samples):
        data = test_dataset[idx]['pixel_values'][None,:,:,:,:].to(device)
        generated_text = model.generate(data,max_new_tokens=config.MAX_LEN,do_sample=True,top_k = 0)
        prediction = (generated_text[0][1:], test_dataset[idx]['labels'].to(device))
        rouge_score = utils.compute_metrics(prediction,inference=True)
        generated_commentary = utils.tokenizer.decode(generated_text[0][1:])
        total_rouge_score += rouge_score['rouge2_fmeasure']
        with open(f"{output_dir}/commentary_final_gpt2.txt",'a') as f:
            f.write(generated_commentary+"\n")
            f.write(f"rouge2 precision: {rouge_score['rouge2_precision']} rouge2_recall: {rouge_score['rouge2_recall']} rouge2_fmeasure: {rouge_score['rouge2_fmeasure']}\n")
    with open(f"{output_dir}/commentary_final_gpt2.txt", 'a') as f:
        f.write(f"average rouge:{total_rouge_score/no_of_test_samples}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, help='Directory where model checkpoints will be saved')
    args = parser.parse_args()
    output_dir = args.output_dir
    train_model(output_dir)