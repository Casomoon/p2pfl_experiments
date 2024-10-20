#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2022 Pedro Guijas Bravo.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

"""Lightning Learner for P2PFL."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import lightning as L
import numpy as np
import torch
from lightning import Trainer
from torch.utils.data import DataLoader

from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.learner import NodeLearner
from p2pfl.learning.p2pfl_model import P2PFLModel
from p2pfl.learning.pytorch.lightning_dataset import PyTorchExportStrategy
from p2pfl.learning.pytorch.lightning_logger import FederatedLogger
from pytorch_lightning.loggers import CSVLogger
from p2pfl.management.logger import logger
from p2pfl_experiments.src.modelling.bert_lightning import BERTLightningModel
from pathlib import Path

torch.set_num_threads(1)


class LightningLearner(NodeLearner):
    """
    Learner with PyTorch Lightning.

    Args:
        model: The model of the learner.
        data: The data of the learner.
        self_addr: The address of the learner.

    """

    def __init__(
        self,
        model: P2PFLModel,
        data: P2PFLDataset,
        self_addr: str = "unknown-node",
    ) -> None:
        """Initialize the learner."""
        self.model = model
        self.data = data
        self.__trainer: Optional[Trainer] = None
        self.epochs = 1
        self.__self_addr = self_addr

        # Start logging
        self.logger = FederatedLogger(self_addr)
        assert isinstance(self.model.model, BERTLightningModel)
        assert isinstance(self.model.model.node_dir, Path)
        self.csv_logger = CSVLogger(save_dir=self.model.model.node_dir, name=self.__self_addr)
        # To avoid GPU/TPU printings
        logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

    def set_model(self, model: Union[P2PFLModel, List[np.ndarray], bytes]) -> None:
        """
        Set the model of the learner.

        Args:
            model: The model of the learner.

        """
        if isinstance(model, P2PFLModel):
            self.model = model
        elif isinstance(model, (list, bytes)):
            self.model.set_parameters(model)

    def get_model(self) -> P2PFLModel:
        """
        Get the model of the learner.

        Returns:
            The model of the learner.

        """
        return self.model

    def set_data(self, data: P2PFLDataset) -> None:
        """
        Set the data of the learner.

        Args:
            data: The data of the learner.

        """
        self.data = data

    def get_data(self) -> P2PFLDataset:
        """
        Get the data of the learner.

        Returns:
            The data of the learner.

        """
        return self.data

    def set_epochs(self, epochs: int) -> None:
        """
        Set the number of epochs.

        Args:
            epochs: The number of epochs.

        """
        self.epochs = epochs

    def __get_pt_model_data(self, splits: list[str]= ["train", "valid"]) -> Tuple[L.LightningModule, DataLoader]:
        # Get Model
        pt_model = self.model.get_model()
        if not isinstance(pt_model, L.LightningModule):
            raise ValueError(f"The model must be a PyTorch Lightning model, the model has type {type(pt_model)}")
        # Get Data
        pt_data: list[DataLoader] = []
        for split in splits: 
            pt_loader = self.data.export(PyTorchExportStrategy, split)
            if not isinstance(pt_loader, DataLoader): 
                raise ValueError("The data must be a PyTorch DataLoader")
            pt_data.append(pt_loader)
        return pt_model, pt_data

    def fit(self) -> None:
        """Fit the model."""
        try:
            logger.info(self.__self_addr, "Starting fit.")
            if self.epochs > 0:
                self.__trainer = Trainer(
                    max_epochs=self.epochs,
                    accelerator="auto",
                    logger=[self.logger, self.csv_logger],
                    enable_checkpointing=False,
                    enable_model_summary=False,
                )
                pt_model, pt_data = self.__get_pt_model_data(["train", "valid"])
                self.__trainer.fit(
                    model=pt_model, 
                    train_dataloaders=pt_data[0],
                    val_dataloaders=pt_data[1],
                    )
                self.__trainer = None
            logger.info(self.__self_addr, "Finished learning.")
            # Set model contribution
            self.model.set_contribution([self.__self_addr], self.data.get_num_samples())

        except Exception as e:
            logger.error(
                self.__self_addr,
                f"Fit error. Something went wrong with pytorch lightning. {e}",
            )
            raise e

    def interrupt_fit(self) -> None:
        """Interrupt the fit."""
        if self.__trainer is not None:
            self.__trainer.should_stop = True
            self.__trainer = None

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model with actual parameters.

        Returns:
            The evaluation results.

        """
        try:
            logger.info(self.__self_addr, "Start evaluation")
            if self.epochs > 0:
                self.__trainer = Trainer(logger=[self.logger, self.csv_logger])
                pt_model, pt_data = self.__get_pt_model_data(["test"])
                results = self.__trainer.test(
                    model=pt_model,
                    dataloaders=pt_data, 
                    verbose=True
                    )[0]
                self.__trainer = None
                # Log metrics
                for k, v in results.items():
                    logger.log_metric(self.__self_addr, k, v)
                logger.info(self.__self_addr, "Finished evaluation" )
                return dict(results)

            else:
                return {}
        except Exception as e:
            logger.error(
                self.__self_addr,
                f"Evaluation error. Something went wrong with pytorch lightning. {e}",
            )
            raise e
