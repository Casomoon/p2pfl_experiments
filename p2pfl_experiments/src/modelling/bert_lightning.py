import torch
import gc 
import torch.nn as nn
import lightning as L 
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score
from typing import Optional, Tuple
from p2pfl.management.logger import logger

from .bert_zoo import get_bert_by_string

class BERTLightningModel(L.LightningModule):
    models = {}
    def __init__(
        self,
        cid: int, 
        model_name: str= "bert", 
        num_labels: int = 2,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
        lr: float = 2e-5,
        seed: Optional[int] = None
    ):
        """Initialize the BERT model."""
        super().__init__()
        # Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        self.cid = cid 
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.module_name = f"BERT_Lightning_{self.cid}"
        self.model = get_bert_by_string(model_name, num_labels)
        use_token_type_ids = True 
        if model_name == "distilbert": use_token_type_ids = False
        self.use_token_type_ids = use_token_type_ids
        self.lr = lr
        # Set up metrics
        self.metric = BinaryAccuracy()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of the BERT models for sentecne classifcation tasks like NLI or CD ."""
        # BERT and MobileBERT
        if token_type_ids is not None: 
            return self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # DistilBERT
        return self.model(input_ids=input_ids, attention_mask=attention_mask)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizer."""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
        {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.lr,
            weight_decay=self.weight_decay
        )
    
        # Learning rate scheduler (linear warmup and decay)
        
        return optimizer

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step of the BERT models."""
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        # assure token type ids dont get passed/ get passed as None based on which model is used
        labels = batch["label"].long()
        # condidtional forward pass 
        if self.use_token_type_ids: 
            token_type_ids = batch["token_type_ids"]
            outputs = self(input_ids=input_ids, attention_mask=attention_mask, token_type_ids = token_type_ids)
        else: outputs = self(input_ids=input_ids, attention_mask=attention_mask) 
        loss = self.loss_fn(outputs.logits, labels)
        preds = torch.argmax(outputs.logits, dim=1)
        metric = self.metric(preds, labels)
        f1 = BinaryF1Score()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_metric", metric, prog_bar=True)

        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step of the BERT models."""
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"].long()
        if self.use_token_type_ids: 
            token_type_ids = batch["token_type_ids"]
            outputs = self(input_ids=input_ids, attention_mask=attention_mask, token_type_ids = token_type_ids)
        else: outputs = self(input_ids=input_ids, attention_mask=attention_mask)
        loss = self.loss_fn(outputs.logits, labels)
        preds = torch.argmax(outputs.logits, dim=1)
        metric = self.metric(preds, labels)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_metric", metric, prog_bar=True)


    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step of the BERT models."""
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"].long()
        unique_labels = torch.unique(labels)
         # Check for invalid labels
        if not torch.all((unique_labels >= 0) & (unique_labels <= 1)):
            logger.info(self.module_name, f"Invalid labels in {unique_labels}.")
            raise ValueError(f"Invalid labels found in batch {batch_idx}: {unique_labels}")
        if self.use_token_type_ids: 
            token_type_ids = batch["token_type_ids"]
            outputs = self(input_ids=input_ids, attention_mask=attention_mask, token_type_ids = token_type_ids)
        else: outputs = self(input_ids=input_ids, attention_mask=attention_mask)
        loss = self.loss_fn(outputs.logits, labels)
        preds = torch.argmax(outputs.logits, dim=1)
        metric = self.metric(preds, labels)
        
        
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_metric", metric, prog_bar=True)
    def on_train_end(self) -> None:
        logger.info(self.module_name, "Training complete. Clearing up VRAM")
        gc.collect()
        torch.cuda.empty_cache()
    def on_validation_end(self) -> None:
        logger.info(self.module_name, "Validation complete. Clearing up VRAM")
        gc.collect()
        torch.cuda.empty_cache()    
    def on_test_end(self) -> None:
        logger.info(self.module_name, "Test complete. Clearing up VRAM")
        gc.collect()
        torch.cuda.empty_cache()  