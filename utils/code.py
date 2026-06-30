import torch
import evaluate


def greedy_decode(model, src, max_len, bos_id, eos_id, device):
    model.eval()
    src = src.to(device)
    src_mask = (src == 0)

    batch_size = src.size(0)
    memory = model.encode(src, src_padding_mask=src_mask)

    ys = torch.full((batch_size, max_len), 0, dtype=torch.long, device=device)
    ys[:, 0] = bos_id

    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for t in range(1, max_len):
        tgt = ys[:, :t]

        out = model.decode(
            tgt,
            memory,
            tgt_padding_mask=(tgt == 0),
            memory_key_padding_mask=src_mask
        )

        next_token = out[:, -1, :].argmax(dim=-1)
        ys[:, t] = next_token

        finished |= next_token.eq(eos_id)

        
        src_lens = (src != 0).sum(dim=1)  
        tgt_lens = t + 1  

        over_limit = (tgt_lens > src_lens * 2 + 5)
        if over_limit.any():
            finished |= over_limit

        if finished.all():
            break

    return ys


# def greedy_decode(model, src, max_len, bos_id, eos_id, device):
#     model.eval()
#     src = src.to(device)
#     src_mask = (src == 0)

#     batch_size = src.size(0)
#     memory = model.encode(src, src_padding_mask=src_mask)


#     ys = torch.full((batch_size, 128), 0, dtype=torch.long, device=device)
#     ys[:, 0] = bos_id


#     finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

#     for t in range(1, max_len):

#         tgt = ys[:, :t]
#         out = model.decode(
#             tgt,
#             memory,
#             tgt_padding_mask=(tgt == 0),
#             memory_key_padding_mask=src_mask
#         )

#         next_token = out[:, -1, :].argmax(dim=-1)  
        
#         ys[:, t] = next_token

#         finished |= next_token.eq(eos_id)


#         if finished.all():
#             break

#     return ys


# def greedy_decode(model, src, max_len, bos_id, eos_id, device):
#     src = src.to(device)
#     src_mask = (src == 0)

#     memory = model.encode(src, src_mask)
#     ys = torch.ones(1, 1).fill_(bos_id).type(torch.long).to(device)
#     memory = memory.to(device)
#     for i in range(max_len-1):
#         tgt = ys[:, :i]
        
#         out = model.decode(
#             tgt,
#             memory,
#             tgt_padding_mask=(tgt == 0),
#             memory_key_padding_mask=src_mask
#         )
#         out = out.transpose(0, 1)
#         prob = model.generator(out[:, -1])
#         _, next_word = torch.max(prob, dim=1)
#         next_word = next_word.item()

#         ys = torch.cat([ys,
#                         torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
#         if next_word == eos_id:
#             break
#     return ys





bleu_metric = evaluate.load("bleu")
sbleu_metric = evaluate.load("sacrebleu") 
meteor_metric = evaluate.load("meteor")




def compute_bleu(pred_tokens, tgt_tokens, tokenizer):
    decoded_preds = [tokenizer.decode(pred.tolist()) for pred in pred_tokens]
    decoded_labels = [[tokenizer.decode(tgt.tolist())] for tgt in tgt_tokens]
    decoded_preds = [d.strip() for d in decoded_preds]
    decoded_labels = [[l[0].strip()] for l in decoded_labels]
    
    bleu = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)["bleu"] * 100
    sbleu = sbleu_metric.compute(predictions=decoded_preds, references=decoded_labels)["score"]
    meteor = meteor_metric.compute(predictions=decoded_preds, references=decoded_labels)['meteor']

    return {"BLEU": round(bleu,4), "SacreBLEU": round(sbleu,4), "METEOR": round(meteor,4)}