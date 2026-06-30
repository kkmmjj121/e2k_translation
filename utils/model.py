import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, dropout, maxlen=5000):
        super().__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        self.pos_embedding = pos_embedding.unsqueeze(0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.pos_embedding[:, :x.size(1), :].to(x.device)
        return self.dropout(x)

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.emb_size)

def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
    return mask

class CustomTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, emb_size=512,
                 nhead=8, num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.pos_enc = PositionalEncoding(emb_size, dropout)

        self.transformer = nn.Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.generator = nn.Linear(emb_size, tgt_vocab_size)

    def encode(self, src, src_padding_mask=None):
        src_emb = self.pos_enc(self.src_tok_emb(src))
        return self.transformer.encoder(
            src_emb,
            src_key_padding_mask=src_padding_mask
        )

    def decode(self, tgt, memory, tgt_padding_mask=None, memory_key_padding_mask=None):
        tgt_emb = self.pos_enc(self.tgt_tok_emb(tgt))
        tgt_mask = generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        out = self.transformer.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        return self.generator(out)