import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import transformers
from kobert_tokenizer import KoBERTTokenizer

from tqdm import tqdm
import os
import numpy as np
import wandb
import json
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models import RetrieverEncoder
from datasets import RetrieverDataset, KorQuadSampler, korquad_collator

# os.environ['CUDA_LAUNCH_BLOCKING'] = "0, 1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

model_name = "retriever" 
save_path = "result/0922_labelsmoothing"
gpu = "cuda:2" ##########
device = torch.device(gpu)
batch_size = 96 ############
lr =1e-5 ############
eps = 1e-8
epoch = 130
valid_every = 979 ##########

seed = 42
torch.manual_seed(seed)


#######################################################################################################
#Data

train_data_path = "/home/nlplab/hdd1/yoo/KorDPR/dataset/KorQuAD_v1.0_train_processed.p"
val_data_path = "/home/nlplab/hdd1/yoo/KorDPR/dataset/KorQuAD_v1.0_dev_processed.p"

train_dataset = RetrieverDataset(train_data_path)
val_dataset =RetrieverDataset(val_data_path)

train_dataloader = DataLoader( 
    dataset=train_dataset.dataset, # 93926개의 (question-id-passage-answer) 
    batch_sampler=KorQuadSampler(train_dataset.dataset, batch_size=batch_size, drop_last=False),
    collate_fn=lambda x: korquad_collator(x, padding_value=train_dataset.pad_token_id), #x: batch
    #num_workers=4,
        )
val_dataloader = DataLoader( 
    dataset=val_dataset.dataset, # 9927개의 (question-id-passage-answer) 
    batch_sampler=KorQuadSampler(val_dataset.dataset, batch_size=batch_size, drop_last=False),
    collate_fn=lambda x: korquad_collator(x, padding_value=val_dataset.pad_token_id), #x: batch
    #num_workers=4,
        )

tokenizer = KoBERTTokenizer.from_pretrained("skt/kobert-base-v1")


#######################################################################################################
#Loss/Acc

class LabelSmoothingCrossEntropy(torch.nn.Module):
    def __init__(self):
        super(LabelSmoothingCrossEntropy, self).__init__()
    def forward(self, x, target, smoothing=0.1):
        confidence=1.-smoothing
        logprobs=torch.nn.functional.log_softmax(x,dim=-1)
        nll_loss=-logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss=nll_loss.squeeze(1)
        smooth_loss=-logprobs.mean(dim=-1)
        loss=confidence*nll_loss+smoothing*smooth_loss
        return loss.mean()
ls_loss=LabelSmoothingCrossEntropy()

def ibn_loss(pred):
    batch_size = pred.size(0)
    #print(pred) # torch.Size([96, 96])
    target = torch.arange(batch_size).to(device)  
    #print(target) # torch.Size([96])
    return ls_loss(pred,target,0.1)
    #return torch.nn.functional.cross_entropy(pred, target) ##### to(device)?
    
def batch_acc(pred):
    batch_size = pred.size(0)
    target = torch.arange(batch_size) 
    return (pred.detach().cpu().max(1).indices == target).sum().float() / batch_size


#######################################################################################################
#model,optimizer,scheduler

model = RetrieverEncoder()
#model.load_state_dict(torch.load("/home/nlplab/hdd1/yoo/KorDPR_retriever/result/0201_base/57.model",map_location='cuda:0'))
model.to(device)

optimizer = Adam(model.parameters(), lr=lr, eps=eps)
#scheduler = transformers.get_linear_schedule_with_warmup(optimizer, 1000,10000) ##########
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=1)
os.makedirs(save_path, exist_ok=True)


#######################################################################################################
#wandb

# wandb.init(
#             project="Kor Retriever",
#             config={
#                 "batch_size": batch_size,
#                 "lr": lr,
#                 "eps": eps,
#                 "num_warmup_steps": 1000, ##########
#                 "num_training_steps": 100000, ##########
#                 "valid_every": 30, ##########
#             },
#         )


#######################################################################################################
#train

def train():
    global_step_cnt = 0
    prev_best = None
    for e in range(epoch):
        for step, batch in enumerate(tqdm(train_dataloader, desc="Epoch {}".format(str(e)))):
            global_step_cnt += 1
            
            model.train()
            optimizer.zero_grad()
            
            ######################################################################

            q, q_mask, _, p, p_mask, _,_ = batch
            q, q_mask, p, p_mask = (
                q.to(device),
                q_mask.to(device),
                p.to(device),
                p_mask.to(device),
            )

            ######################################################################
            
            q_emb= model(q, q_mask, "query")  
            p_emb = model(p, p_mask, "passage")  
            
            pred = torch.matmul(q_emb, p_emb.T) # bs*bs 
                
            loss =ibn_loss(pred)
            acc = batch_acc(pred)
            
            ######################################################################
            
            loss.backward()
            
            optimizer.step()
            #scheduler.step()
            
            ######################################################################
            
            log = {
                "epoch": e,
                "step": step,
                "global_step": global_step_cnt,
                "train_step_loss": loss.cpu().item(),
                "current_lr": lr, #float(scheduler.get_last_lr()[0]),  
                "step_acc": acc,
            }
            if global_step_cnt % valid_every == 0:
                val_dict = validation()
                #log.update(val_dict)
                print(val_dict)
                if (prev_best is None or val_dict["val_loss"] < prev_best):  
                    torch.save(model.state_dict(), os.path.join(save_path, '{}.model'.format(e)))
                    scheduler.step(val_dict["val_loss"])
                    print("best epoch:",e)
                    print("best epoch:",e)
                    print("best epoch:",e)
            #wandb.log(log)
            print(log)


#######################################################################################################
#validation

def validation():
    
    model.eval()  
    
    loss_list = []
    sample_cnt = 0
    val_acc = 0
    
    with torch.no_grad():
        for batch in val_dataloader:
            
            q, q_mask, _, p, p_mask,_,_ = batch
            q, q_mask, p, p_mask = (
                q.to(device),
                q_mask.to(device),
                p.to(device),
                p_mask.to(device),
            )
            
            ######################################################################
            
            q_emb = model(q, q_mask, "query")  
            p_emb = model(p, p_mask, "passage") 
            
            pred = torch.matmul(q_emb, p_emb.T)  
            
            loss =ibn_loss(pred)
            step_acc = batch_acc(pred)
            
            ######################################################################

            batch_size = q.size(0)
            sample_cnt += batch_size
            val_acc += step_acc * batch_size
            loss_list.append(loss.cpu().item() * batch_size)
            
    return {
        "val_loss": np.array(loss_list).sum() / float(sample_cnt),
        "val_acc": val_acc / float(sample_cnt),
    }


#######################################################################################################
#main

train()