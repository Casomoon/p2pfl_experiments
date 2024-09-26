from torch.utils.data import Dataset, DataLoader
from p2pfl.management.logger import logger
import pytorch_lightning as pl
import torch

class NLIDataset(Dataset): 
    num_classes = 2 
    def __init__(self,cid: int, phase: str, data: list[dict], train: bool = False) -> None:
        assert phase in ["train", "val", "test"]
        self.phase = phase
        self.cid = cid
        self.module_name = f"NLIDataset_{phase}_{self.cid}"
        self.training = train
        self.data: list[dict] = data
        self.label_sanity_check()
    
    def label_sanity_check(self,): 
        # in order to prevent the very big negative number in test data loader do sanity check 
        labels_unique = set()
        for i, element in enumerate(self.data):
            label = element.get("label")
            assert type(label) == int, f"Label at index {i}: in set {self.module_name} is not an integer but {label} has type {type(label)}" 
            labels_unique.add(label)
        assert labels_unique == {0,1}, f"Unexpected labels found: {labels_unique}."

        
    def __len__(self) -> int: 
        return len(self.data)
    
    def __getitem__(self, index) -> dict:
        row = self.data[index]
        encoding = row["encoded"]
        # Prepare x input 
        input = {k : v for k,v in encoding.items()}
        cd_label = row.get("label")
        if not self.training: 
            return input
        if cd_label<0 or cd_label >=self.num_classes: 
            raise ValueError(f"Label {cd_label} is out of range (0 to {self.num_classes - 1})")
        input["label"] = torch.tensor(cd_label).long()
        return input
    
class NLIDataModule(pl.LightningDataModule):
    def __init__(
                self, 
                cid: int, 
                train_loader: DataLoader, 
                val_loader: DataLoader, 
                test_loader: DataLoader
                ):
        super().__init__()
        self.cid = cid
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader 
    
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
        return self.train_loader

    def val_dataloader(self):
        """
        Returns the validation dataloaders for each client as a list.
        """
        return self.val_loader

    def test_dataloader(self):
        """
        Returns the global test dataloader.
        """
        return self.test_loader
    