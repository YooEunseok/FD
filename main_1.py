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
from models import MODEL_DICT, RankerEncoder, RetrieverEncoder
from datasets import RetrieverDataset, KorQuadSampler, korquad_collator


model_name = "retriever" 
save_path = "result/tmp"
gpu = "cuda:2" ##########
device = torch.device(gpu)
batch_size = 42 ############
lr = 1e-5 ############
eps = 1e-8
epoch = 100
valid_every = 100 ##########

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
    batch_sampler=KorQuadSampler(val_dataset.dataset, batch_size=100, drop_last=False),
    collate_fn=lambda x: korquad_collator(x, padding_value=val_dataset.pad_token_id), #x: batch
    #num_workers=4,
        )

tokenizer = KoBERTTokenizer.from_pretrained("skt/kobert-base-v1")


#######################################################################################################
#Loss/Acc

def ibn_loss(pred):
    batch_size = pred.size(0)
    #print(pred) # torch.Size([96, 96])
    target = torch.arange(batch_size).to(device)  
    #print(target) # torch.Size([96])
    return torch.nn.functional.cross_entropy(pred, target) ##### to(device)?

def knowledge_distillation_loss(logits, teacher_logits):
    batch_size = logits.size(0)
    #print(pred) # torch.Size([96, 96])
    target = torch.arange(batch_size).to(device)  
    #print(target) # torch.Size([96])
    
    alpha = 0.1
    T = 10
        
    student_loss = torch.nn.functional.cross_entropy(input=logits, target=target)
    distillation_loss = torch.nn.KLDivLoss(reduction='batchmean') \
        (torch.nn.functional.log_softmax(logits/T, dim=1), torch.nn.functional.softmax(teacher_logits/T, dim=1)) * (T * T)
    total_loss =  alpha*student_loss + (1-alpha)*distillation_loss

    return total_loss

def batch_acc(pred):
    batch_size = pred.size(0)
    target = torch.arange(batch_size) 
    return (pred.detach().cpu().max(1).indices == target).sum().float() / batch_size


#######################################################################################################
#model,optimizer,scheduler

teacher_model = RetrieverEncoder()
teacher_model.load_state_dict(torch.load("/home/nlplab/hdd1/yoo/KorDPR_retriever/result/tmp/0.model"))
teacher_model.to(device)

model = MODEL_DICT[model_name]()
model.to(device)

optimizer = Adam(model.parameters(), lr=lr, eps=eps)
#scheduler = transformers.get_linear_schedule_with_warmup(optimizer, 1000,10000) ##########
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=1)
os.makedirs(save_path, exist_ok=True)


#######################################################################################################
#wandb

wandb.init(
            project="KorDPR_retriever",
            config={
                "batch_size": batch_size,
                "lr": lr,
                "eps": eps,
                "num_warmup_steps": 1000, ##########
                "num_training_steps": 100000, ##########
                "valid_every": 30, ##########
            },
        )


#######################################################################################################
#train

def train():
    global_step_cnt = 0
    prev_best = None
    for e in range(epoch):
        for step, batch in enumerate(tqdm(train_dataloader, desc="Epoch {}".format(str(e)))):
            #print(len(batch[0]))
            #assert()
            global_step_cnt += 1
            
            model.train()
            teacher_model.eval()
            optimizer.zero_grad()
            
            ######################################################################

            q, q_mask, p_id, p, p_mask, _,_ = batch
            q, q_mask, p, p_mask = (
                q.to(device),
                q_mask.to(device),
                p.to(device),
                p_mask.to(device),
            )  

            ######################################################################

            q_emb = model(q, q_mask, "query")  
            p_emb = model(p, p_mask, "passage")  
            
            pred = torch.matmul(q_emb, p_emb.T) # bs*bs 
            
            t_q_emb =teacher_model(q, q_mask, "query")  
            t_p_emb =teacher_model(p, p_mask, "passage")  
                    
            teacher_pred = torch.matmul(t_q_emb, t_p_emb.T) # bs*bs 
                    
            loss = knowledge_distillation_loss(pred, teacher_pred)
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
                log.update(val_dict)
                if (prev_best is None or val_dict["val_loss"] < prev_best):  
                    torch.save(model.state_dict(), os.path.join(save_path, '{}.model'.format(e)))
                    scheduler.step(val_dict["val_loss"])
            wandb.log(log)


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
            
            loss = ibn_loss(pred)
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