from utils import paths
paths.TOKENIZER_ROOT.mkdir(parents=True, exist_ok=True)
import pandas as pd
from collections import Counter
df = pd.read_csv(str(paths.DATA_CSV))

src = df['en'].astype(str).tolist()
tgt = df['ko'].astype(str).tolist()





counter = Counter()
counter2 = Counter()
for line in src:
    counter.update(line.split())

t = 5

rare_words = {w for w, c in counter.items() if c < t}

for line in tgt:
    counter2.update(line.split())

rare_words2 = {w for w, c in counter2.items() if c < t}

# print(rare_words)
def replace_rare(line, rare_words):
    return " ".join([w if w not in rare_words else "<unk>" for w in line.split()])

ps_en = [replace_rare(line, rare_words) for line in src]
ps_ko = [replace_rare(line, rare_words2) for line in tgt]

with open(str(paths.TOKENIZER_ROOT / "corpus_en.txt"), 'w', encoding= 'utf-8') as f:
    for line in ps_en:
        f.write(line.strip() +'\n')

f.close()
with open(str(paths.TOKENIZER_ROOT / "corpus_ko.txt"), 'w', encoding= 'utf-8') as f:
    for line in ps_ko:
        f.write(line.strip() +'\n')
f.close()


print(len(rare_words))
print(len(rare_words2))

import sentencepiece as spm

spm.SentencePieceTrainer.Train(
    input=str(paths.TOKENIZER_ROOT / "corpus_en.txt"),
    model_prefix=str(paths.TOKENIZER_ROOT / "corpus_en"),
    vocab_size=16000,     
    model_type='bpe',      
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
)

spm.SentencePieceTrainer.Train(
    input=str(paths.TOKENIZER_ROOT / "corpus_ko.txt"),
    model_prefix=str(paths.TOKENIZER_ROOT / "corpus_ko"),
    vocab_size=16000,     
    model_type='bpe',      
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
)