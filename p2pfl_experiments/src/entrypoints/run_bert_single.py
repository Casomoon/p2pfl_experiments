import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
from ..modelling.bert_lightning import BERTLightningModel
from ..modelling.nli_data_load import NLIParser
from ..modelling.nli_pl_wrapper import NLIDataModule
from torch.utils.data import Dataset, DataLoader
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryRecall, BinaryPrecision
import transformers
import numpy as np
import lightning as L
from lightning import Trainer
from lightning.pytorch.loggers import CSVLogger
from pathlib import Path
import torch
import gc 
from p2pfl.learning.dataset.p2pfl_dataset import DataExportStrategy, P2PFLDataset
from p2pfl.learning.pytorch.lightning_dataset import PyTorchExportStrategy
import random
root = Path(__file__).resolve().parents[2]
csv_save = root/"logs/single_bert"
mnli_data_path = root/"data"/"multinli_1.0"
model = "BERT"



class TestSetEvaluator(L.Callback):
    def __init__(self, test_dataloader):
        super().__init__()
        self.test_dataloader = test_dataloader
        # Define metrics inside the callback
        self.test_acc = BinaryAccuracy()
        self.test_f1 = BinaryF1Score()
        self.test_recall = BinaryRecall()
        self.test_precision = BinaryPrecision()

    def on_validation_epoch_end(self, trainer, pl_module):
        try:
            print("Evaluating test set...")
            device = pl_module.device
            self.test_acc.to(device)
            self.test_f1.to(device)
            self.test_recall.to(device)
            self.test_precision.to(device)
            # Reset metrics before evaluation
            self.test_acc.reset()
            self.test_f1.reset()
            self.test_recall.reset()
            self.test_precision.reset()

            

            # Iterate over the test dataloader
            for batch in self.test_dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                if pl_module.use_token_type_ids:
                    token_type_ids = batch.get("token_type_ids", None)
                    if token_type_ids is not None:
                        token_type_ids = token_type_ids.to(device)
                    outputs = pl_module(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                else:
                    outputs = pl_module(input_ids=input_ids, attention_mask=attention_mask)

                preds = torch.argmax(outputs.logits, dim=1)

                # Update metrics with this batch
                self.test_acc.update(preds, labels)
                self.test_f1.update(preds, labels)
                self.test_recall.update(preds, labels)
                self.test_precision.update(preds, labels)

            # Compute final metrics after processing all test batches
            test_acc_val = self.test_acc.compute().item()
            test_f1_val = self.test_f1.compute().item()
            test_recall_val = self.test_recall.compute().item()
            test_precision_val = self.test_precision.compute().item()

            # Log final metrics
            pl_module.log("test_acc", test_acc_val, on_epoch=True, prog_bar=True)
            pl_module.log("test_f1", test_f1_val, on_epoch=True, prog_bar=True)
            pl_module.log("test_recall", test_recall_val, on_epoch=True, prog_bar=True)
            pl_module.log("test_precision", test_precision_val, on_epoch=True, prog_bar=True)
        except Exception as e:
            print(f"Error during test evaluation: {e}")

def set_deterministic_training(seed: int):
    # Python & NumPy
    random.seed(seed)
    np.random.seed(seed)
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # PyTorch >=1.8
    torch.use_deterministic_algorithms(True)
    # Force deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # transformers  
    transformers.set_seed(seed)
  

def main(): 
    set_deterministic_training(420)
    torch.set_float32_matmul_precision("medium")
    csv_logger = CSVLogger(save_dir=csv_save, name="SingleBertTraining")
    data_parser = NLIParser(mnli_data_path, 1, [1.0], model_name="bert", batch_size=1, shuffle = True, overall_cut=0.0)
    data_module:P2PFLDataset = data_parser.get_non_iid_split()[0]
    train = data_module.export(PyTorchExportStrategy,"train")
    val = data_module.export(PyTorchExportStrategy,"valid")
    test = data_module.export(PyTorchExportStrategy,"test")
    print(f"Number of test batches: {len(test)}")
  
    csv_save.mkdir()
    bert_model = BERTLightningModel(cid=0, model_name="bert", num_labels=2, lr=2e-5, base_dir=csv_save)
    gc.collect()    
    # Initialize the PyTorch Lightning Trainer
    test_set_evaluator = TestSetEvaluator(test_dataloader=test)
    trainer = Trainer(
        max_epochs=4,  # You can adjust the number of epochs
        accelerator="gpu",  # Use GPU if available
        logger = csv_logger,
        num_sanity_val_steps=0,
        callbacks=[test_set_evaluator]
    )
    print(trainer.callbacks)
    # Train the model
    trainer.fit(bert_model,train_dataloaders=train, val_dataloaders=val)
    