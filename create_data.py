import torch
from torch import tensor as T
import pickle
from kobert_tokenizer import KoBERTTokenizer
import json
from indexers import DenseFlatIndexer
from models import RetrieverEncoder
from datasets import DATASET_DICT, RetrieverDataset, KorQuadSampler, korquad_collator
from glob import glob
from conifg import DATAPATH_DICT
from models import MODEL_DICT


gpu = "cuda:2"
device = torch.device(gpu)

class KorDPRRetriever:
    def __init__(self, model, dataset, index, device='cuda:0'):
        self.model = model.to(device)
        self.device = device
        self.tokenizer = dataset.tokenizer
        self.index = index
        
    def retrieve(self, query: str, k: int = 100):
        """주어진 쿼리에 대해 가장 유사도가 높은 passage를 반환합니다."""
        self.model.eval()  # 평가 모드
        tok = self.tokenizer.batch_encode_plus([query])
        with torch.no_grad():
            out = self.model(T(tok["input_ids"]).to(self.device), T(tok["attention_mask"]).to(self.device), "query")
        result = self.index.search_knn(query_vectors=out.cpu().numpy(), top_docs=k)

        #print("query: ",query)
        # 원문 가져오기
        passages = []
        for idx, sim in zip(*result[0]):
            path = get_passage_file([idx])
            if not path:
                print(f"No single passage path for {idx}")
                continue
            with open(path, "rb") as f:
                passage_dict = pickle.load(f)
            #print(f"passage : {passage_dict[idx]}, sim : {sim}")
            p=tokenizer.encode(passage_dict[idx][1:-1])
            passages.append((passage_dict[idx], sim))
        return passages


def get_passage_file(p_id_list) -> str:
    """passage id를 받아서 해당되는 파일 이름을 반환합니다."""
    target_file = None
    p_id_max = max(p_id_list)
    p_id_min = min(p_id_list)
    for f in glob("processed_passages/*.p"):
        s, e = f.split("/")[1].split(".")[0].split("-")
        s, e = int(s), int(e)
        if p_id_min >= s and p_id_max <= e:
            target_file = f
    return target_file


if __name__ == "__main__":
    
    model = RetrieverEncoder()
    model.load_state_dict(torch.load("/home/nlplab/hdd1/yoo/KorDPR_retriever/result/1212_origin/130.model"))
    model.eval()
    
    train_dataset = RetrieverDataset("/home/nlplab/hdd1/yoo/KorDPR/dataset/KorQuAD_v1.0_train_processed.p")
    valid_dataset = RetrieverDataset("/home/nlplab/hdd1/yoo/KorDPR/dataset/KorQuAD_v1.0_dev_processed.p")
    
    index = DenseFlatIndexer() 
    index.deserialize(path="/home/nlplab/hdd1/yoo/KorDPR/1212/origin/epoch_130")
    retriever = KorDPRRetriever(model=model, dataset=valid_dataset, index=index) #####
    
    model_name = "retriever" 
    
    tokenizer = KoBERTTokenizer.from_pretrained("skt/kobert-base-v1")
    
    file_path = "./ranker_data_valid_1212_130_100.json" #####
    json_data = {}
    json_data['data'] = []
    temp=[]
    
    print(len(valid_dataset.dataset)) #####
    cnt=0
    good=0
    for i in range(len(valid_dataset.dataset)): ##### #len(train_dataset.dataset)
        item=valid_dataset.dataset[i] #####
        query=tokenizer.decode(item[0][1:-1]).strip()
        idx=item[1]
        passage=tokenizer.decode(item[2][1:-1])
        answer=tokenizer.decode(item[3][1:-1])       
        
        q=tokenizer.encode(query) 

        neg_passages=retriever.retrieve(query=query, k=100)

        neg=[]
        no_cnt=0
        #print("len(neg_passages)",len(neg_passages))
        flag=0
        for j in range(len(neg_passages)):
            temp_set=set(temp)
            #count=0
            if answer in neg_passages[j][0] and flag==0:
                good+=1
                flag=1
                
            '''
            if answer in neg_passages[j][0]:
                no_cnt+=1
                neg.append(neg_passages[j-1][0]) # 앞 neg passage 한번 더 넣어주기
            else:           
                neg.append(neg_passages[j][0])
            '''
            neg.append(neg_passages[j][0])
            
        #print("no_cnt:",no_cnt) # 최대가 1인데 나중에 다시 확인해보기
        
        json_data['data'].append({
            'query':query ,
            'id':idx,
            'passage':passage,
            'topK_passages': neg,
            'answer': answer
        })
        
        cnt+=1
        print(good)
        print(cnt)
        print("acc:",good/cnt)

    with open(file_path, 'w') as outfile:
        json.dump(json_data, outfile, indent=4)        