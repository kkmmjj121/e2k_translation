import os
import torch
from transformers import TranslationPipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import pandas as pd
import sentencepiece as sp
from utils.code import greedy_decode
from utils.model import CustomTransformer
from sklearn.model_selection import train_test_split

import evaluate




seed = 55

df = pd.read_csv('en2ko.csv')

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

best_model = AutoModelForSeq2SeqLM.from_pretrained("/NasData/home/kmj/test/term/model/en2ko/checkpoint-64110").to(device)
best_tokenizer = AutoTokenizer.from_pretrained("/NasData/home/kmj/test/term/model/en2ko/checkpoint-64110")


translator = TranslationPipeline(model=best_model, tokenizer=best_tokenizer)

sample_data = df.sample(100, random_state=seed)

df_sorted = df.loc[df['en'].str.len().sort_values(ascending=True).index]

df_sample = df_sorted[150000:1501000]
sbleu_metric = evaluate.load("sacrebleu") 

train_data, temp = train_test_split(df, test_size=0.2, random_state=seed)
val_data, test_data = train_test_split(temp, test_size=0.5, random_state=seed)

test_data = test_data.loc[test_data['en'].str.len().sort_values(ascending=True).index]

class CustomTranslationPipeline:
    def __init__(self, model, en_tokenizer, ko_tokenizer,device):
        self.model = model
        self.en_tokenizer = en_tokenizer
        self.ko_tokenizer = ko_tokenizer
        self.device = device

    def __call__(self, text, max_length=128):
        tokens = self.en_tokenizer.encode(text, out_type=int)
        src = torch.tensor(tokens).unsqueeze(0).to(self.device)
        generated = greedy_decode(
            self.model,
            src,
            max_len=max_length,
            bos_id=self.en_tokenizer.bos_id(),
            eos_id=self.en_tokenizer.eos_id(),
            device=self.device
        )
        pred_text = self.ko_tokenizer.decode(generated[0].tolist())
        return {"translation_text": pred_text}

state_dict = torch.load("/NasData/home/kmj/test/term/model/self/last_model.pt")

# state_dict2 = torch.load("/NasData/home/kmj/test/term/model/self/final_final_model.pt")

en_sp_tokenizer = sp.SentencePieceProcessor()
en_sp_tokenizer.load("corpus_en.model")
ko_sp_tokenizer = sp.SentencePieceProcessor()
ko_sp_tokenizer.load('corpus_ko.model')

# en_sp_tokenizer2 = sp.SentencePieceProcessor()
# en_sp_tokenizer2.load("en.model")
# ko_sp_tokenizer2 = sp.SentencePieceProcessor()
# ko_sp_tokenizer2.load('ko.model')



src_vocab_size = en_sp_tokenizer.get_piece_size()
tgt_vocab_size = ko_sp_tokenizer.get_piece_size()

# src_vocab_size2 = en_sp_tokenizer2.get_piece_size()
# tgt_vocab_size2 = ko_sp_tokenizer2.get_piece_size()

model = CustomTransformer(src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size)
model.load_state_dict(state_dict)
model.to(device)


# model2 = CustomTransformer(src_vocab_size=src_vocab_size2, tgt_vocab_size=tgt_vocab_size2)
# model2.load_state_dict(state_dict2)
# model2.to(device)


translator2 = CustomTranslationPipeline(model, en_sp_tokenizer, ko_sp_tokenizer, device)
# translator3 = CustomTranslationPipeline(model2, en_sp_tokenizer2, ko_sp_tokenizer2, device)


# import nltk.translate.bleu_score as bleu

sbleu_metric = evaluate.load("sacrebleu")
meteor_metric = evaluate.load('meteor')
t_m = 0
t_b = 0
c = 0
count = 0
for i, row in test_data.iterrows():
    src_text = row["en"]
    tgt_text = row["ko"]
    if len(src_text) < 300:
        continue
    if len(src_text) > 400:
        break
    # pred = translator(src_text, max_length=128, src_lang = "eng_Latn", tgt_lang = "kor_Hang")[0]['translation_text']
    pred2 = translator2(src_text, max_length=128)['translation_text']
    # print()
    
    bleu_score = sbleu_metric.compute(predictions=[pred2], references=[[tgt_text]])['score']
    meteor_score = meteor_metric.compute(predictions=[pred2], references=[tgt_text])['meteor']
    
    t_m += meteor_score
    t_b += bleu_score
    c += 1
    count += 1
    if c == 1000:
        break
    # if bleu_score > 10:
        # print(f"METEOR: {meteor_score:.4f}")
        # print(f"SacreBLEU: {bleu_score:.2f}")
        # print(f"Source: {src_text}")
        # print(f"Target: {tgt_text}")
        # # print(f"Prediction: {pred}")
        # print(f'Prediction2: {pred2}')
    

print(f'METEOR: {t_m/count :.4f}')
print(f'SacreBLEU: {t_b/count :.2f}')
print(count)

# for i, row in df_sample.iterrows():
#     src_text = row["en"]
#     tgt_text = row["ko"]
#     # pred = translator(src_text, max_length=128, src_lang = "eng_Latn", tgt_lang = "kor_Hang")[0]['translation_text']
#     pred2 = translator2(src_text, max_length=128)['translation_text']
#     pred3 = translator3(src_text, max_length=128)['translation_text']
#     bleu_score = bleu.sentence_bleu(list(map(lambda ref: ref.split(), [tgt_text])),pred2.split())*100
#     bleu_score2 = bleu.sentence_bleu(list(map(lambda ref: ref.split(), [tgt_text])),pred3.split())*100
#     if bleu_score > 1:
#         print()
#         print(f"Source: {src_text}")
#         print(f"Target: {tgt_text}")
#         # print(f"Prediction: {pred}")
#         print(f'Prediction2: {pred2}')
        
        
        
#         print(f'{ bleu_score : .4f}')

#         print(f'Prediction3: {pred3}')

#         print(f'{ bleu_score2 : .4f}')

    # sbleu = sbleu_metric.compute(predictions=pred2.split(), references=[[tgt_text.split()]][0])["score"]
    # print(round(sbleu,4))


# print(translator2("Each nation's goods markets must be in balance, and both nations must be taken into account simultaneously." , max_length=128)['translation_text'])