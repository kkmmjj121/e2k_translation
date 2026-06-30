from utils.model import CustomTransformer
import pandas as pd
import sentencepiece as sp
import torch
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from utils.dataset import TranslationDataset
import torch.nn as nn

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seed = 42


sp_tokenizer = sp.SentencePieceProcessor()
sp_tokenizer.load("corpus_ko.model")

s = sp_tokenizer.encode('그러나 현실에서는 적어도 단기에는 환율의 비전가 현상이 자주 관측되고, 국가의 지정학적 특성에 따라 이자율에 리스크 프리미엄이 존재하여 이론적 예측과는 달리 금융시장에서 이자율평형조건이 잘 성립하지 않는다.')
# print(sp_tokenizer.decode(sp_tokenizer.encode('만약 이미 채식을 하는데도 고기 먹고 싶은 마음이 그치지 않으면, 매일 아침 지장보살 멸정업진언을 21번 외우고, 불전에 올린 물 한잔을 마십니다.')))

for i in s:
    print(sp_tokenizer.Decode(i))





# print(sp_tokenizer)
# df = pd.read_csv("en2ko.csv")
# # df = df[:100]

# unk_id = sp_tokenizer.unk_id()

# total_tokens = 0
# unk_tokens = 0

# for sent in df['ko'].tolist():
#     ids = sp_tokenizer.encode(sent, out_type=int)
#     total_tokens += len(ids)
#     unk_tokens += (torch.tensor(ids) == unk_id).sum().item()

# print(total_tokens, unk_tokens)
# print("UNK 비율:", unk_tokens / total_tokens * 100, "%")