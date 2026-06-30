import pandas as pd
import os
import sentencepiece as spm
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


df = pd.read_csv('en2ko.csv')


# en = df['ko'].tolist()
# en2 = []
# for i in en:
#     k = i+'\n'
#     en2.append(k)









# with open('ko.txt', 'w') as f:
#     for i in en2:
#         f.write(i)
# f.close()







# sp = spm.SentencePieceProcessor(model_file='ko.model')
# print(sp.get_piece_size())
# print(sp.pad_id(), sp.unk_id(), sp.bos_id(), sp.eos_id())


# # print(sp.encode('.'))

spm.SentencePieceTrainer.Train(
    input='ko.txt',   
    model_prefix='ko',        
    vocab_size=48000,            
    model_type='bpe',            
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
    user_defined_symbols="<special1>,<special2>"
)




