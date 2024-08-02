from calendar import EPOCH
from turtle import forward
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import json
import pandas as pd

from conifg import DATAPATH_DICT
from models import MODEL_DICT
from datasets import DATASET_DICT
from utils import set_seed, torch_MCC, torch_ACC

assert MODEL_DICT.keys() == DATASET_DICT.keys(), "Model and dataset not matched"

#load model path
save_path = "result/tmp"
epoch = 0
setting_json = json.load(open(os.path.join(save_path, 'setting.json'),"r"))

#set config and hyperparameter
task = setting_json["task"]
model_name = setting_json["model_name"]
pretrain_model_path = setting_json["pretrain_model_path"] #korean bert https://huggingface.co/klue/bert-base
save_path = "result/tmp"
gpu = "cuda:0"
device = torch.device(gpu)
batch_size = 128
max_length = 30
EPOCH = 30

#set seed
seed = 42
set_seed(seed)

test_data_path = DATAPATH_DICT[task]["test"]
#load dataset and dataloader
test_dataset = DATASET_DICT[model_name](test_data_path, pretrain_model_path, max_length)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

class COLA_dataset(Dataset):
    def __init__(self, file_path, pretrain_model_path, max_len):
        super().__init__()
        self.data = pd.read_csv(file_path, sep="\t")
        self.tokenizer = AutoTokenizer.from_pretrained(pretrain_model_path)
        self.max_len = max_len
        print(self.data)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sentData = self.data.iloc[index,:]
            
        tok = self.tokenizer(sentData['sentence'], padding="max_length", max_length=self.max_len, truncation=True)

        input_ids=torch.LongTensor(tok["input_ids"])
        token_type_ids=torch.LongTensor(tok["token_type_ids"])
        attention_mask=torch.LongTensor(tok["attention_mask"])
        
        label = None
        if 'acceptability_label' in sentData.keys():
            label = sentData['acceptability_label']
            
        return input_ids, token_type_ids, attention_mask, label









#from torch.utils.data import Dataset
from kobert_tokenizer import KoBERTTokenizer
from torch.nn.utils.rnn import pad_sequence
from torch import tensor
from torch.utils.data import Dataset

import os
import json
import pickle
import torch


class RetrieverDataset(Dataset):
    def __init__(self, file_path):
        super().__init__()
        self.korquad_processed_path = file_path
        self.data_tuples = []
        self.tokenizer = KoBERTTokenizer.from_pretrained("skt/kobert-base-v1")
        self.pad_token_id = self.tokenizer.get_vocab()["[PAD]"]
        self.sep_token_id = self.tokenizer.get_vocab()["[SEP]"]
        
        with open(self.korquad_processed_path, "rb") as f:
            self.tokenized_tuples = pickle.load(f)
        with open('/home/nlplab/hdd1/yoo/KorDPR_retriever/ranker_data.json', "rb") as f:
            self.neg_passages = json.load(f)
        
    def __len__(self):
        return len(self.tokenized_tuples)
                
    def __getitem__(self, index):
        data = self.tokenized_tuples[index]
        query=data[0]
        id=data[1]
        passage=data[2]
        answer=data[3]
        
        while 2 in passage: # 숫자 2가 지워 지는 일은 없는지 확인 필요
            passage.remove(2)

        #print(query)
        #print(passage)
        query.extend(passage)
        pos_pair=query
        #print(input)
        
        data1=self.neg_passages
        ##### query 별 top k 저장해놓은 파일 불러오기 #####
        # 인코딩 되어있게 할건가?
        #topK_passage=data1[str(id)]
        try:
            topK_passage=data1[str(id)]
        except:
            topK_passage=data1["1337502"]
        print(topK_passage[0])
        for i in range(len(topK_passage)):
            while 2 in topK_passage[i]: # 숫자 2가 지워 지는 일은 없는지 확인 필요
                topK_passage[i].remove(2)
        print(topK_passage)
        assert 0
        
        q = pad_sequence(query, batch_first=True, padding_value=self.pad_token_id)
        q_attn_mask = (query != self.pad_token_id).long()
        p = pad_sequence(passage, batch_first=True, padding_value=self.pad_token_id)
        p_attn_mask = (passage != self.pad_token_id).long()
        
        neg_p=[]
        neg_p_attn_mask=[]
        for i in range(len(topK_passage)):
            neg_p.append(pad_sequence(passage, batch_first=True, padding_value=self.pad_token_id))
            neg_p_attn_mask.append((passage != self.pad_token_id).long())
            
        return q, q_attn_mask, p, p_attn_mask, neg_p, neg_p_attn_mask


DATASET_DICT = {
    "retriever": RetrieverDataset,
}

