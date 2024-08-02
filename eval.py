import torch
import pickle
from collections import defaultdict
from tqdm import tqdm

#from main import val_data_path, device, batch_size
from indexers import DenseFlatIndexer
from models import RetrieverEncoder
from datasets import RetrieverDataset, KorQuadSampler, korquad_collator
from utils import get_passage_file


gpu = "cuda:0"
device = torch.device(gpu)
batch_size=128

class KorDPRRetriever:
    def __init__(self, model, valid_dataset, index, batch_size= batch_size, device=device):
        self.model = model.to(device)
        self.device = device
        self.tokenizer = valid_dataset.tokenizer
        self.batch_size = batch_size
        
        self.valid_loader = torch.utils.data.DataLoader(
            dataset=valid_dataset.dataset,
            batch_sampler=KorQuadSampler(valid_dataset.dataset, batch_size=self.batch_size, drop_last=False),
            collate_fn=lambda x: korquad_collator(x, padding_value=valid_dataset.pad_token_id),
            num_workers=4,
        )
        self.index = index

    def val_top_k_acc(self, k=[5] + list(range(10,101,10))):        
        #k=[5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        #print(k) #[5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        k_max = max(k)
        sample_cnt = 0
        retr_cnt = defaultdict(int)
        with torch.no_grad():
            for batch in tqdm(self.valid_loader, desc='valid'):
                q, q_mask, p_id, p, p_mask, a, a_mask = batch
                
                q, q_mask = (
                    q.to(self.device),
                    q_mask.to(self.device),
                )
                
                q_emb = self.model(q, q_mask, "query") 
                
                result = self.index.search_knn(query_vectors=q_emb.cpu().numpy(), top_docs=k_max) 
                # [([967695, 718599], array([44.343082, 44.31025 ], dtype=float32))]
                for ((pred_idx_lst, _),_, _a , _a_mask) in zip(result, p_id, a, a_mask): # k번 도는 loop
                    
                    a_len = _a_mask.sum()
                    _a = _a[:a_len]                    
                    _a = _a[1:-1]                    
                    _a_txt = self.tokenizer.decode(_a).strip()
                    
                    docs = [pickle.load(open(get_passage_file([idx]),'rb'))[idx] for idx in pred_idx_lst]

                    for _k in k:
                        if _a_txt in ' '.join(docs[:_k]): 
                            retr_cnt[_k] += 1

                bsz = q.size(0)
                sample_cnt += bsz
        retr_acc = {_k:float(v) / float(sample_cnt) for _k,v in retr_cnt.items()}
        return retr_acc


if __name__ == "__main__":

    model = RetrieverEncoder()
    model.load_state_dict(torch.load("/home/nlplab/hdd1/yoo/KorDPR_retriever/result/0922_selfregularization/120.model"))
    #model.load("checkpoint/2050iter_model.pt")
    model.eval()
    
    # valid_dataset = RetrieverDataset("dataset/KorQuAD_v1.0_dev.json")
    valid_dataset = RetrieverDataset("/home/nlplab/hdd1/yoo/KorDPR/dataset/KorQuAD_v1.0_dev_processed.p")
    
    
    index = DenseFlatIndexer()
    index.deserialize(path="/home/nlplab/hdd1/yoo/KorDPR/0922_selfregularization/120")
    
    retriever = KorDPRRetriever(model=model, valid_dataset=valid_dataset, index=index)
    
    retr_acc = retriever.val_top_k_acc()
    print(retr_acc)
