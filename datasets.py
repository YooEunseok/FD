from kobert_tokenizer import KoBERTTokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import BatchSampler, RandomSampler
from torch import tensor
import pickle
import json
import torch


class RetrieverDataset:
    def __init__(self, file_path):
        super().__init__()
        self.korquad_processed_path = file_path
        self.tokenizer = KoBERTTokenizer.from_pretrained("skt/kobert-base-v1")
        self.pad_token_id = self.tokenizer.get_vocab()["[PAD]"]
        
        with open(self.korquad_processed_path, "rb") as f:
            self.tokenized_tuples = pickle.load(f) # 93926/9955
                
    @property        
    def dataset(self):
        return self.tokenized_tuples #(question-id-passage-answer)
                

class KorQuadSampler(BatchSampler):
    def __init__(self, data_source, batch_size, drop_last = False):
        sampler = RandomSampler(data_source, replacement=False) #, generator=generator )
        super(KorQuadSampler, self).__init__(sampler=sampler, batch_size=batch_size, drop_last=drop_last)
        self.tokenizer = KoBERTTokenizer.from_pretrained("skt/kobert-base-v1")

    def __iter__(self): 
        sampled_p_id = []
        sampled_idx = []
        for idx in self.sampler: 
            item = self.sampler.data_source[idx] 
            if item[1] in sampled_p_id:
                continue  # 만일 같은 answer passage가 이미 뽑혔다면 pass
            sampled_idx.append(idx) # 실제 index
            sampled_p_id.append(item[1]) # 해당 itesm의 p_id
            if len(sampled_idx) >= self.batch_size: 
                yield sampled_idx # yield는 return과 다르게 결과값을 여러번에 나누어서 반환해줌
                sampled_p_id = []
                sampled_idx = []
        if len(sampled_idx) > 0 and not self.drop_last: # batch 다 묶어주고 남은것
            yield sampled_idx


def korquad_collator(batch, padding_value): 

    batch_q = pad_sequence(
        [tensor(e[0]) for e in batch], batch_first=True, padding_value=padding_value
    ) # 40~60 정도
    batch_q_attn_mask = (batch_q != padding_value).long()
    batch_p_id = tensor([e[1] for e in batch])[:, None]
    batch_p = pad_sequence(
        [tensor(e[2]) for e in batch], batch_first=True, padding_value=padding_value
    ) # 100~140 정도
    batch_p_attn_mask = (batch_p != padding_value).long()
    batch_a = pad_sequence(
        [tensor(e[3]) for e in batch], batch_first=True, padding_value=padding_value
    )
    batch_a_attn_mask = (batch_a != padding_value).long()
    
    return (batch_q, batch_q_attn_mask, batch_p_id, batch_p, batch_p_attn_mask,batch_a,batch_a_attn_mask)


DATASET_DICT = {
    "retriever": RetrieverDataset,
}

