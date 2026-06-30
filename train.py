import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader
from utils.seed import seed_everything
from utils.dataset import CustomDataset
from sklearn.model_selection import train_test_split
import argparse
import warnings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq, EarlyStoppingCallback
import evaluate
import wandb


# 환경 설정
warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 파라미터 설정
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type= int, default= 42)
args = parser.parse_args()

# 시드 설정
seed = args.seed
seed_everything(seed)


# 데이터 불러오기 & 데이터 분할
data = pd.read_csv('/NasData/home/kmj/test/term/en2ko.csv')

train_data, temp = train_test_split(data, test_size=0.2, random_state=seed)
val_data, test_data = train_test_split(temp, test_size=0.5, random_state=seed)




# 모델 & 토크나이저 불러오기

tokenizer = AutoTokenizer.from_pretrained("NHNDQ/nllb-finetuned-en2ko")
model = AutoModelForSeq2SeqLM.from_pretrained("NHNDQ/nllb-finetuned-en2ko")



# tokenizer.src_lang = "eng_Latn"
# tokenizer.tgt_lang = "kor_Hang"


# 데이터셋으로 저장
train_ds = CustomDataset(train_data, tokenizer, 128)
val_ds   = CustomDataset(val_data, tokenizer, 128)
test_ds  = CustomDataset(test_data, tokenizer, 128)


metric = evaluate.load('sacrebleu')


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result



data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model="NHNDQ/nllb-finetuned-en2ko")


wandb.init(
    project="finetune-translation",
    name=f"run_seed{seed}_model_NHNDQ-nllb",
    config={
        "seed": seed,
        "model_name": "NHNDQ/nllb-finetuned-en2ko",
        "batch_size": 16,
        "learning_rate": 2e-5,
        "num_epochs": 50,
    }
)


training_args = Seq2SeqTrainingArguments(
    output_dir="model/en2ko",
    eval_strategy="epoch",                   
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="bleu",
    greater_is_better=True,
    num_train_epochs=50,              
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    predict_with_generate=True,
    logging_steps=1000,
    save_total_limit=3,
    fp16=True,                       
    report_to="wandb",
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)



trainer.train()

results = trainer.evaluate(eval_dataset=test_ds)
print(results)
