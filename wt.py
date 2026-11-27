from utils import paths
paths.TOKENIZER_ROOT.mkdir(parents=True, exist_ok=True)
import pandas as pd
import os
import sentencepiece as spm
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("E2K_CUDA", "0"))


df = pd.read_csv(str(paths.DATA_CSV))


# en = df['ko'].tolist()
# en2 = []
# for i in en:
#     k = i+'\n'
#     en2.append(k)









# with open(str(paths.TOKENIZER_ROOT / "ko.txt"), 'w') as f:
#     for i in en2:
#         f.write(i)
# f.close()







# sp = spm.SentencePieceProcessor(model_file=str(paths.TOKENIZER_ROOT / "ko.model"))
# print(sp.get_piece_size())
# print(sp.pad_id(), sp.unk_id(), sp.bos_id(), sp.eos_id())


# # print(sp.encode('.'))

spm.SentencePieceTrainer.Train(
    input=str(paths.TOKENIZER_ROOT / "ko.txt"),
    model_prefix=str(paths.TOKENIZER_ROOT / "ko"),
    vocab_size=48000,            
    model_type='bpe',            
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
    user_defined_symbols="<special1>,<special2>"
)




