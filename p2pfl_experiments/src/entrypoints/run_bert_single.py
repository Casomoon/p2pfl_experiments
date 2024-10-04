import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from ..modelling.bert_lightning import BERTLightningModel
from ..modelling.nli_data_load import NLIParser
from ..modelling.nli_pl_wrapper import NLIDataModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pathlib import Path
import torch
import gc 

root = Path(__file__).resolve().parents[2]
csv_save = root/"logs"
mnli_data_path = root/"data"/"multinli_1.0"
model = "BERT"

def main(): 
    torch.set_float32_matmul_precision("medium")
    csv_logger = CSVLogger(save_dir=csv_save, name="SingleBertTraining")
    data_parser = NLIParser(mnli_data_path, 1, [1.0], batch_size=1, shuffle = True)
    data_modules = data_parser.get_non_iid_split()
    assert len(data_modules) == 1
    single_data_module = data_modules[0]
    bert_model = BERTLightningModel(cid=0, model_name="bert", num_labels=2, lr=2e-5)
    gc.collect()    
    # Initialize the PyTorch Lightning Trainer
    trainer = Trainer(
        max_epochs=4,
        accumulate_grad_batches=2,  # You can adjust the number of epochs
        gpus=torch.cuda.device_count(),  # Use GPU if available
        logger = csv_logger
    )

    # Train the model
    trainer.fit(bert_model, single_data_module)

    # Optionally, test the model on the global test set
    trainer.test(bert_model, dataloaders=single_data_module.test_dataloader())