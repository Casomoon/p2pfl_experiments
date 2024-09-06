from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup
import torch
import os
from logging import Logger
from numpy import ndarray
from torch.utils.data import DataLoader
from torch.optim import AdamW

class BERT_Peer():
    def __init__(self, 
                cid : int, 
                logger : Logger, 
                trainloader : DataLoader,
                valloader : DataLoader) -> None:
        self.cid = cid
        self.logger = logger
        self.train = trainloader
        self.val = valloader
        self.model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels = 2)
        # Prepare optimizer and schedule (linear warmup and decay)
        self.optimizer = AdamW(self.model.parameters(), lr=os.environ.get("LR"), eps=os.environ.get("EPSILON"))
        self.scheduler = get_linear_schedule_with_warmup(self.opti, num_warmup_steps=0, num_training_steps=total_steps)
        # Move model to GPU if available
        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available")
        
        self.device = torch.device("cuda")
        self.logger.info(f"Device {self.device} is used.")

    def get_parameters(self, ) -> torch.List[ndarray]:
        return 
    def set_parameters(self,):
        return
    def fit(self,):
        self.logger.info()
        self.model.to(self.device)
        epochs = os.environ.get("EPOCHS")
        for epoch in range(epochs): 
            for batch in self.train: 
                self.logger.info(batch)
                self.logger.info(type(batch))
                pair_token_ids = pair_token_ids.to(self.device)
            mask_ids = mask_ids.to(self.device)
            seg_ids = seg_ids.to(self.device)
            labels = y.to(self.device)

            loss, prediction = self.model(pair_token_ids,
                                     token_type_ids=seg_ids,
                                     attention_mask=mask_ids,
                                     labels=labels).values()

            self.loss.backward()
            self.optimizer.step()

            #total_train_loss += loss.item()
            #total_train_acc += acc.item()
            #it += 1  
    
    def free_gpu_from_client_model(self): 
        self.model.to(torch.device("cpu"))
        del self.model
        

    def evaluate():pass