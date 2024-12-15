import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from ..modelling.bert_lightning import BERTLightningModel
from ..modelling.nli_data_load import NLIParser
from ..modelling.nli_pl_wrapper import NLIDataModule
from torch.utils.data import Dataset, DataLoader
import numpy as np
import lightning as L
from lightning import Trainer
from lightning.pytorch.loggers import CSVLogger
from pathlib import Path
import torch
import gc 
from p2pfl.learning.dataset.p2pfl_dataset import DataExportStrategy, P2PFLDataset
from p2pfl.learning.pytorch.lightning_dataset import PyTorchExportStrategy
root = Path(__file__).resolve().parents[2]
csv_save = root/"logs/single_bert"
mnli_data_path = root/"data"/"multinli_1.0"
model = "BERT"

from lightning.pytorch.callbacks import Callback

class TestSetEvaluator(L.Callback):
    def __init__(self, test_dataloader):
        self.test_dataloader = test_dataloader

    def on_validation_epoch_end(self, trainer, pl_module):
        try:
            print("Evaluating test set...")
            total_metrics = {metric_name: 0.0 for metric_name in pl_module.metrics.keys()}
            num_batches = 0

            for batch in self.test_dataloader:
                input_ids = batch["input_ids"].to(pl_module.device)
                attention_mask = batch["attention_mask"].to(pl_module.device)
                labels = batch["label"].to(pl_module.device)

                if pl_module.use_token_type_ids:
                    token_type_ids = batch.get("token_type_ids", None).to(pl_module.device)
                    outputs = pl_module(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                else:
                    outputs = pl_module(input_ids=input_ids, attention_mask=attention_mask)

                preds = torch.argmax(outputs.logits, dim=1)

                # Accumulate metrics
                for metric_name, metric_fn in pl_module.metrics.items():
                    metric_value = metric_fn(preds, labels)
                    total_metrics[metric_name] += metric_value.item()

                num_batches += 1

            # Compute averages
            for metric_name, total in total_metrics.items():
                avg_metric = total / num_batches
                print(f"Logging test_{metric_name}: {avg_metric}")
                pl_module.log(f"test_{metric_name}", avg_metric, on_epoch=True, prog_bar=True)
        except Exception as e:
            print(f"Error during test evaluation: {e}")
def set_deterministic_training(seed: int ): 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def main(): 
    set_deterministic_training(420)
    torch.set_float32_matmul_precision("medium")
    csv_logger = CSVLogger(save_dir=csv_save, name="SingleBertTraining")
    data_parser = NLIParser(mnli_data_path, 1, [1.0], model_name="bert", batch_size=1, shuffle = True, overall_cut=0.)
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
    