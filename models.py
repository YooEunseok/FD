import torch
from torch import nn
import transformers
from transformers import BertModel


class RetrieverEncoder(nn.Module):
    def __init__(self):
        super(RetrieverEncoder, self).__init__()
        self.passage_encoder = BertModel.from_pretrained("skt/kobert-base-v1")
        self.query_encoder = BertModel.from_pretrained("skt/kobert-base-v1")
        #self.hidden_size=768

    def forward(self, input_ids, att_mask, type):
        if type == "passage":
            return self.passage_encoder(
                input_ids=input_ids, attention_mask=att_mask
            ).pooler_output
        else:
            return self.query_encoder(
                input_ids=input_ids, attention_mask=att_mask
            ).pooler_output


MODEL_DICT = {
    "retriever": RetrieverEncoder,
}