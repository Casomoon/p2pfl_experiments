import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy
from transformers import BertForSequenceClassification
from typing import Optional, Tuple

class BERTLightningModel(pl.LightningModule):
    def __init__(
        self,
        model_name: str = 'bert-base-uncased',
        num_labels: int = 2,
        metric: type[Accuracy] = Accuracy,
        lr: float = 2e-5,
        seed: Optional[int] = None
    ):
        """Initialize the BERT model."""
        super().__init__()
        # Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.lr = lr
        # Set up metrics
        self.metric = metric(task="binary")
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass of the BERT model."""
        return self.model(input_ids=input_ids, attention_mask=attention_mask)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizer."""
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step of the BERT model."""
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        
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
        labels = batch["labels"]

        outputs = self(input_ids=input_ids, attention_mask=attention_mask)
        loss = self.loss_fn(outputs.logits, labels)
        preds = torch.argmax(outputs.logits, dim=1)
        metric = self.metric(preds, labels)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_metric", metric, prog_bar=True)

        return loss

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step of the BERT model."""
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        outputs = self(input_ids=input_ids, attention_mask=attention_mask)
        loss = self.loss_fn(outputs.logits, labels)
        preds = torch.argmax(outputs.logits, dim=1)
        metric = self.metric(preds, labels)

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_metric", metric, prog_bar=True)

        return loss