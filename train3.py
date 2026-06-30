import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from utils.seed import seed_everything
from sklearn.model_selection import train_test_split
import sentencepiece as sp
import evaluate
import math
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import argparse
from tqdm import tqdm
from utils.model import CustomTransformer
from utils.dataset import TranslationDataset
from utils.code import compute_bleu, greedy_decode




# 파라미터 설정
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type= int, default= 42)
parser.add_argument('--N', type= int, default=6)
parser.add_argument('--d_model', type = int, default=512)
parser.add_argument('--dff', type = int, default=2048)
parser.add_argument('--head', type=int, default=8)
parser.add_argument('--cuda', type=str, default='1')
parser.add_argument('--batch', type=int, default=128)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--name', type = str, default='final_final_model.pt')
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--warm', type=float, default=0.1)
parser.add_argument('--sc', type= int, default=1)
parser.add_argument('--p', type=int, default=10)
parser.add_argument('--drop', type= float, default=0.3)
args = parser.parse_args()

seed = args.seed
seed_everything(seed)

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sp_tokenizer_en = sp.SentencePieceProcessor()
sp_tokenizer_en.load("en.model")
sp_tokenizer_ko = sp.SentencePieceProcessor()
sp_tokenizer_ko.load('ko.model')

src_vocab_size = sp_tokenizer_en.get_piece_size()
tgt_vocab_size = sp_tokenizer_ko.get_piece_size()

pad_id = sp_tokenizer_en.pad_id()




df = pd.read_csv("en2ko.csv")
# df = df[:100]
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=seed)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

train_dataset = TranslationDataset(train_df, sp_tokenizer_en, sp_tokenizer_ko)
val_dataset = TranslationDataset(val_df, sp_tokenizer_en, sp_tokenizer_ko)
test_dataset = TranslationDataset(test_df, sp_tokenizer_en, sp_tokenizer_ko)


def collate_fn(batch, pad_id=0):
    src_batch, tgt_batch = zip(*batch)
    src_batch = nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=pad_id)
    tgt_batch = nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=pad_id)
    return src_batch, tgt_batch

train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, collate_fn=lambda x: collate_fn(x, pad_id=0))
# val_loader   = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=lambda x: collate_fn(x, pad_id=0))
test_loader  = DataLoader(test_dataset, batch_size=args.batch, shuffle=False, collate_fn=lambda x: collate_fn(x, pad_id=0))

val_dataset_sorted = sorted(val_dataset, key=lambda x: len(x[0]))
val_loader = DataLoader(val_dataset_sorted, batch_size=args.batch, shuffle=False, collate_fn=collate_fn)
model = CustomTransformer(src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size, emb_size=args.d_model,
                 nhead=args.head, num_encoder_layers=args.N, num_decoder_layers=args.N,
                 dim_feedforward=args.dff, dropout=args.drop).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.98), eps=1e-9,)   
criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
state_dict = torch.load("/NasData/home/kmj/test/term/model/self/final_model.pt")
model.load_state_dict(state_dict)
def train_one_epoch(model, dataloader, optimizer, criterion, scheduler, device, log_interval=100):
    model.train()
    total_loss = 0
    
    for i, (src, tgt) in enumerate(dataloader, 1):
        src, tgt = src.to(device), tgt.to(device)
        tgt_input = tgt[:, :-1]  
        tgt_labels = tgt[:, 1:]  

        optimizer.zero_grad()


        memory = model.encode(src, src_padding_mask=(src == 0))

        outputs = model.decode(
            tgt_input,
            memory,
            tgt_padding_mask=(tgt_input == 0),
            memory_key_padding_mask=(src == 0)
        )

        loss = criterion(
            outputs.reshape(-1, outputs.size(-1)),
            tgt_labels.reshape(-1)
        )

        loss.backward()
        optimizer.step()
        sc.step()

        total_loss += loss.item()
        

        if i % log_interval == 0:
            print(f"Iteration {i}/{len(dataloader)}, Loss: {total_loss/i:.4f}")

    return total_loss / len(dataloader)







num_epochs = args.epoch
total_steps = len(train_loader) * num_epochs
warmup_steps = len(train_loader) * args.warm
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)


scheduler2 = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

s_type = args.sc
if s_type == 1:
    sc = scheduler
else:
    sc = scheduler2

best_bleu = 0
patience = args.p
c = 0

model_save = '/NasData/home/kmj/test/term/model/self'
model_name = args.name
for epoch in range(num_epochs):

    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scheduler, device)
    total_loss = 0

    model.eval()
    all_pred_tokens = []
    all_tgt_tokens = []
    with torch.no_grad():
        loop = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}", ncols=100)
        for src, tgt in loop:
            src, tgt = src.to(device), tgt.to(device)

            generated = greedy_decode(
                model,
                src,
                max_len=128,
                bos_id=sp_tokenizer_ko.bos_id(),
                eos_id=sp_tokenizer_ko.eos_id(),
                device=device
            )

            all_pred_tokens.extend(generated[:, 1:].cpu())
            all_tgt_tokens.extend(tgt[:, 1:].cpu())
            tgt_input = tgt[:, :-1]  
            tgt_labels = tgt[:, 1:]  
            memory = model.encode(src, src_padding_mask=(src == 0))

            outputs = model.decode(
            tgt_input,
            memory,
            tgt_padding_mask=(tgt_input == 0),
            memory_key_padding_mask=(src == 0)
            )
            loss = criterion(
            outputs.reshape(-1, outputs.size(-1)),
            tgt_labels.reshape(-1)
            )
            total_loss += loss.item()

    val_bleu = compute_bleu(all_pred_tokens, all_tgt_tokens, sp_tokenizer_ko)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Val BLEU: {val_bleu}")
    print(f'val_loss : {total_loss/ len(val_loader)}')
    sbleu = val_bleu['SacreBLEU']
    if sbleu > best_bleu:
        best_bleu = sbleu
        c = 0
        torch.save(model.state_dict(), os.path.join(model_save, model_name))
    else:
        c += 1
        if c >= patience:
            print('early stopping')
            break


model.eval()
all_pred_tokens = []
all_tgt_tokens = []

with torch.no_grad():
    for src, tgt in test_loader:
        generated = greedy_decode(model, src, max_len=128, bos_id=sp_tokenizer_ko.bos_id(), eos_id=sp_tokenizer_ko.eos_id(), device=device)
        all_pred_tokens.extend(generated[:, 1:].cpu())
        all_tgt_tokens.extend(tgt.cpu())

test_bleu = compute_bleu(all_pred_tokens, all_tgt_tokens, sp_tokenizer_ko)
print(f"Test BLEU: {test_bleu}")









# N	d_model	d_ff	h	d_k	d_v	P_drop	e
# 6	512		2048	8	64	64	0.1		0.1		25.8
# 			4096								26.2	90
# 			1024		128	128					26.0	168

# 6	1024	4096	16			0.3				26.4	213

# 12 1024    4096   16          0.1                     600?     256000