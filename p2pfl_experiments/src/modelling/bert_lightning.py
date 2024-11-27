import torch
import gc 
import torch.nn as nn
import lightning as L 
import torch
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
        model_name: str= "bert", 
        num_labels: int = 2,
        weight_decay: float = 0.01,
        lr: float = 2e-5,
        seed: Optional[int] = None,
        base_dir: Path = None
    ):
        """Initialize the BERT model."""
        
        super().__init__()
        # Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        self.cid = cid 
        self.weight_decay = weight_decay
        self.module_name = f"BERT_Lightning_{self.cid}"
        self.model : PreTrainedModel = get_bert_by_string(model_name, num_labels)
        use_token_type_ids = True 
        if model_name == "distilbert": use_token_type_ids = False
        self.use_token_type_ids = use_token_type_ids
        self.lr = lr
        # Set up metrics
        self.metrics = {
            "acc": BinaryAccuracy(),
            "f1": BinaryF1Score(),
            "recall": BinaryRecall(),
            "precision": BinaryPrecision()
            }
        self.loss_fn = nn.CrossEntropyLoss()
        # each entry contains one processed batch -> one for each test_step call
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
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps*0.1)
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
        self.log("train_loss", loss, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        for metric_name, metric_fn in self.metrics.items(): 
            metric_fn = metric_fn.to(self.device)
            metric_value = metric_fn(preds,labels)
            self.log(   
                f"train_{metric_name}", 
                metric_value, 
                prog_bar=True, 
                logger = True, 
                on_epoch = True, 
                on_step=False
                )
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
        self.log("val_loss", loss, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        preds = torch.argmax(outputs.logits, dim=1)
        for metric_name, metric_fn in self.metrics.items():
            metric_fn = metric_fn.to(self.device)
            metric_value = metric_fn(preds, labels)
            self.log(
                f"val_{metric_name}", 
                metric_value, 
                prog_bar = True, 
                logger = True,
                on_epoch=True, 
                on_step=False
                )
        return loss



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
        self.log("test_loss", loss, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        for metric_name, metric_fn in self.metrics.items(): 
            metric_fn = metric_fn.to(self.device)
            metric_value = metric_fn(preds, labels)
            self.log(
                f"test_{metric_name}",
                metric_value,
                prog_bar= True,
                logger=True,
                on_epoch=True,
                on_step=False
                )
        self.test_step_outputs.append({"preds": preds, "labels": labels})
        return loss
    
    def on_test_epoch_end(self) -> None:
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