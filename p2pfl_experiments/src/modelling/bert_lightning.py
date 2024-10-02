import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAccuracy
from transformers import BertForSequenceClassification
from typing import Optional, Tuple
from p2pfl.management.logger import logger

class BERTLightningModel(pl.LightningModule):
    def __init__(
        self,
        cid: int, 
        model_name: str = 'bert-base-uncased',
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
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.lr = lr
        # Set up metrics
        self.metric = BinaryAccuracy()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass of the BERT model."""
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
        """Training step of the BERT model."""
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]
        
        outputs = self(input_ids=input_ids, attention_mask=attention_mask)
        loss = self.loss_fn(outputs.logits, labels)
        preds = torch.argmax(outputs.logits, dim=1)
        metric = self.metric(preds, labels)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_metric", metric, prog_bar=True)

        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step of the BERT model."""
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]

        outputs = self(input_ids=input_ids, attention_mask=attention_mask)
        loss = self.loss_fn(outputs.logits, labels)
        preds = torch.argmax(outputs.logits, dim=1)
        metric = self.metric(preds, labels)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_metric", metric, prog_bar=True)


    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step of the BERT model."""
        
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]
        unique_labels = torch.unique(labels)
         # Check for invalid labels
        if not torch.all((unique_labels >= 0) & (unique_labels <= 1)):
            logger.info(self.module_name, f"Invalid labels in {unique_labels}.")
            raise ValueError(f"Invalid labels found in batch {batch_idx}: {unique_labels}")
        #if len(unique_labels) < 2:
        #   
        #   #print(f"Batch {batch_idx} has only one class: {unique_labels}")
        #   return  # Skip metric computation or handle accordingly

        outputs = self(input_ids=input_ids, attention_mask=attention_mask)
        loss = self.loss_fn(outputs.logits, labels)
        preds = torch.argmax(outputs.logits, dim=1)
        metric = self.metric(preds, labels)
        
        # Logging for debugging
        #logger.info(self.module_name,f"Batch {batch_idx} - Unique labels: {unique_labels}")
        #logger.info(self.module_name,f"Labels min/max: {labels.min()}/{labels.max()}")
        #logger.info(self.module_name,f"Labels dtype: {labels.dtype}")
        #logger.info(self.module_name,f"Labels device: {labels.device}")
        #logger.info(self.module_name,f"Preds device: {preds.device}")
        #logger.info(self.module_name,f"Preds dtype: {preds.dtype}")
        #logger.info(self.module_name,f"Preds shape: {preds.shape}")
        #logger.info(self.module_name,f"Labels shape: {labels.shape}")
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_metric", metric, prog_bar=True)
