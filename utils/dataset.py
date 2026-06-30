import torch
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings("ignore")
import sentencepiece as spm

class CustomDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.df = df
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        src = self.df['en'].iloc[idx]
        tgt = self.df['ko'].iloc[idx]
        return self.tokenizer(src, text_target=tgt, max_length=self.max_len, truncation=True)


# <s>
# </s>


sp = spm.SentencePieceProcessor(model_file='en2ko.model')
bos_id = sp.bos_id()
eos_id = sp.eos_id()
def tensor_transform(token_ids):
    return torch.cat((torch.tensor([bos_id]),
                      torch.tensor(token_ids),
                      torch.tensor([eos_id])))

class TranslationDataset(Dataset):
    def __init__(self, df, tokenizer_en, tokenizer_ko , max_len=128):
        self.src = df['en'].tolist()
        self.tgt = df['ko'].tolist()
        self.tokenizer_en = tokenizer_en
        self.tokenizer_ko = tokenizer_ko
        self.max_len = max_len

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        
        src_tokens = self.tokenizer_en.encode(self.src[idx], out_type=int)[:self.max_len]
        tgt_tokens = self.tokenizer_ko.encode(self.tgt[idx], out_type=int)[:self.max_len]
        
        tgt = tensor_transform(tgt_tokens)
        return torch.tensor(src_tokens, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)
