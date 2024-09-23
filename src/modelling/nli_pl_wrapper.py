from .nli_data_load import NLIParser 
from torch.utils.data import Dataset, DataLoader
from p2pfl.management.logger import logger
import pytorch_lightning as pl
import torch

class NLIDataset(Dataset): 
    def __init__(self, cid: int, data: list[dict], train: bool = False) -> None:
        self.cid = cid 
        self.module_name = f"NLIDataset {self.cid}"
        self.training = train
        self.data = data
        
    def __len__(self) -> int: 
        return len(self.data)
    
    def __getitem__(self, index) -> dict:
        row = self.data[index]
        encoding = row["encoded"]
        # Prepare x input 
        input = {k : v.squeeze(0) for k,v in encoding.items()}
        if not self.training: 
            return input
        cd_label : int = row["label"]
        input["label"] = torch.tensor(cd_label).long()
        return input
    
class NLIDataModule(pl.LightningDataModule):
    def __init__(
                self, 
                parser: NLIParser,
                cid: int, 
                niid: bool = True
                ):
        super().__init__()
        self.parser = parser
        self.cid = cid
        self.niid = niid
        self.train_loaders: list[DataLoader]
        self.val_loaders: list[DataLoader]
        self.global_test: DataLoader
        split_dataset = self.parser.get_non_iid_split()
        self.train_loaders, self.val_loaders, self.global_test = split_dataset
    
    def setup(self, stage: str = None):
        """
        Called at the beginning of the fit, test, or predict process.
        """
        # No need to do anything here, the data has already been prepared in the NLIParser module
        pass
    
    def train_dataloader(self):
        """
        Returns the training dataloaders for each client as a list.
        """
        return self.train_loaders[self.cid]

    def val_dataloader(self):
        """
        Returns the validation dataloaders for each client as a list.
        """
        return self.val_loaders[self.cid]

    def test_dataloader(self):
        """
        Returns the global test dataloader.
        """
        return self.global_test