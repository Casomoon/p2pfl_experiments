import torch
import gc 
import torch.nn as nn
import lightning as L 
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryRecall, BinaryPrecision
from typing import Optional, Tuple
from transformers import get_linear_schedule_with_warmup
from transformers.modeling_utils import PreTrainedModel
from p2pfl.management.logger import logger
from pathlib import Path
from .bert_zoo import get_bert_by_string
from ..stats.result_visualization import plot_confusion_matrix

class BERTLightningModel(L.LightningModule):
    def __init__(
        self,
        cid: int, 
        model_name: str = "bert", 
        num_labels: int = 2,
        weight_decay: float = 0.01,
        lr: float = 2e-5,
        seed: Optional[int] = None,
        base_dir: Path = None
    ):
        """Initialize the BERT model."""
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        self.cid = cid 
        self.weight_decay = weight_decay
        self.module_name = f"BERT_Lightning_{self.cid}"
        self.model: PreTrainedModel = get_bert_by_string(model_name, num_labels)
        use_token_type_ids = True 
        if model_name == "distilbert": 
            use_token_type_ids = False
        self.use_token_type_ids = use_token_type_ids
        self.lr = lr
        
        # Metrics for each phase (set compute_on_step=False to aggregate over entire epoch)
        self.train_acc = BinaryAccuracy(compute_on_step=False)
        self.train_f1 = BinaryF1Score(compute_on_step=False)
        self.train_recall = BinaryRecall(compute_on_step=False)
        self.train_precision = BinaryPrecision(compute_on_step=False)

        self.val_acc = BinaryAccuracy(compute_on_step=False)
        self.val_f1 = BinaryF1Score(compute_on_step=False)
        self.val_recall = BinaryRecall(compute_on_step=False)
        self.val_precision = BinaryPrecision(compute_on_step=False)

        self.test_acc = BinaryAccuracy(compute_on_step=False)
        self.test_f1 = BinaryF1Score(compute_on_step=False)
        self.test_recall = BinaryRecall(compute_on_step=False)
        self.test_precision = BinaryPrecision(compute_on_step=False)

        self.loss_fn = nn.CrossEntropyLoss()
        self.test_step_outputs: list[dict] = []
        assert isinstance(base_dir, Path)
        assert base_dir.exists()
        self.setup_model_results_dir(base_dir)
        self.round = 0

    def setup_model_results_dir(self, base_dir: Path):
        node_dir = base_dir/f"node_{self.cid}"
        assert not node_dir.exists()
        node_dir.mkdir()
        self.node_dir = node_dir

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        if token_type_ids is not None: 
            return self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return self.model(input_ids=input_ids, attention_mask=attention_mask)

    def configure_optimizers(self) -> torch.optim.Optimizer:
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
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * 0.1)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"].long()

        if self.use_token_type_ids: 
            token_type_ids = batch["token_type_ids"]
            outputs = self(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        else:
            outputs = self(input_ids=input_ids, attention_mask=attention_mask)

        loss = self.loss_fn(outputs.logits, labels)
        preds = torch.argmax(outputs.logits, dim=1)

        # Update metrics
        self.train_acc.update(preds, labels)
        self.train_f1.update(preds, labels)
        self.train_recall.update(preds, labels)
        self.train_precision.update(preds, labels)

        # Log loss per step
        self.log("train_loss", loss, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        return loss

    def on_train_epoch_end(self):
        # Compute and log metrics once per epoch
        self.log("train_acc", self.train_acc.compute(), prog_bar=True)
        self.log("train_f1", self.train_f1.compute(), prog_bar=True)
        self.log("train_recall", self.train_recall.compute(), prog_bar=True)
        self.log("train_precision", self.train_precision.compute(), prog_bar=True)

        # Reset metrics
        self.train_acc.reset()
        self.train_f1.reset()
        self.train_recall.reset()
        self.train_precision.reset()

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"].long()

        if self.use_token_type_ids: 
            token_type_ids = batch["token_type_ids"]
            outputs = self(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        else:
            outputs = self(input_ids=input_ids, attention_mask=attention_mask)

        loss = self.loss_fn(outputs.logits, labels)
        preds = torch.argmax(outputs.logits, dim=1)

        # Update metrics
        self.val_acc.update(preds, labels)
        self.val_f1.update(preds, labels)
        self.val_recall.update(preds, labels)
        self.val_precision.update(preds, labels)

        self.log("val_loss", loss, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        return loss

    def on_validation_epoch_end(self):
        # Compute and log metrics once per epoch
        self.log("val_acc", self.val_acc.compute(), prog_bar=True)
        self.log("val_f1", self.val_f1.compute(), prog_bar=True)
        self.log("val_recall", self.val_recall.compute(), prog_bar=True)
        self.log("val_precision", self.val_precision.compute(), prog_bar=True)

        # Reset metrics
        self.val_acc.reset()
        self.val_f1.reset()
        self.val_recall.reset()
        self.val_precision.reset()

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"].long()
        unique_labels = torch.unique(labels)
        if not torch.all((unique_labels >= 0) & (unique_labels <= 1)):
            logger.info(self.module_name, f"Invalid labels in {unique_labels}.")
            raise ValueError(f"Invalid labels found in batch {batch_idx}: {unique_labels}")

        if self.use_token_type_ids: 
            token_type_ids = batch["token_type_ids"]
            outputs = self(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        else:
            outputs = self(input_ids=input_ids, attention_mask=attention_mask)

        loss = self.loss_fn(outputs.logits, labels)
        preds = torch.argmax(outputs.logits, dim=1)

        # Update metrics
        self.test_acc.update(preds, labels)
        self.test_f1.update(preds, labels)
        self.test_recall.update(preds, labels)
        self.test_precision.update(preds, labels)

        self.log("test_loss", loss, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        self.test_step_outputs.append({"preds": preds, "labels": labels})
        return loss

    def on_test_epoch_end(self):
        # Compute and log metrics once per epoch
        self.log("test_acc", self.test_acc.compute(), prog_bar=True)
        self.log("test_f1", self.test_f1.compute(), prog_bar=True)
        self.log("test_recall", self.test_recall.compute(), prog_bar=True)
        self.log("test_precision", self.test_precision.compute(), prog_bar=True)

        self.test_acc.reset()
        self.test_f1.reset()
        self.test_recall.reset()
        self.test_precision.reset()

        all_preds = torch.cat([out["preds"] for out in self.test_step_outputs])
        all_labels = torch.cat([out["labels"] for out in self.test_step_outputs])
        np_preds = all_preds.cpu().numpy()
        np_labels = all_labels.cpu().numpy()
        plot_confusion_matrix(np_preds, np_labels, self.node_dir, self.cid, self.round)
        self.cur_model_to_disk()
        self.round += 1
        self.test_step_outputs.clear()

    def on_train_end(self) -> None:
        logger.info(self.module_name, "Training complete. Clearing up VRAM")
        self.clear_vram()

    def on_validation_end(self) -> None:
        logger.info(self.module_name, "Validation complete. Clearing up VRAM")
        self.clear_vram()

    def on_test_end(self) -> None:
        logger.info(self.module_name, "Test complete. Clearing up VRAM")
        self.clear_vram()

    def clear_vram(self):
        torch.cuda.empty_cache()
        gc.collect()

    def cur_model_to_disk(self):
        model_name = f"{self.module_name}_{self.round}"
        model_path = self.node_dir/f"{model_name}.pth"
        torch.save(self.model.state_dict(), model_path)