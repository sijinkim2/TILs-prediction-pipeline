import os
from pathlib import Path
from models import *
import torch
from sources.pltrain import SSLearner
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from datamodule.simclr_datamodule import SimCLRDataModule
#import os # os.getpid()

class Application(object):

    def __init__(self, cfg):

        self.cfg = cfg
        self.tb_logger = TensorBoardLogger(save_dir=self.cfg.get('logging_params')['save_dir'],
                                      name= self.cfg.get('logging_params')['name'])

        Path(f"{self.tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
        Path(f"{self.tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)

    def run(self):

        dm = SimCLRDataModule(self.cfg.get("datamodule"))
        dm.setup()

        model = ssl_models[self.cfg.get('model')['name']](self.cfg.get('model'))
        pl_model = SSLearner(model, self.cfg.get('pl_train'))

        # For reproducibility
        seed_everything(self.cfg['pl_train']['manual_seed'])

        runner = Trainer(logger= self.tb_logger,
                         callbacks=[
                             LearningRateMonitor(),
                             ModelCheckpoint(save_top_k=2,
                                             dirpath=os.path.join(self.tb_logger.log_dir, "checkpoints"),
                                             monitor="val_loss",
                                             save_last=True)],
                         **self.cfg.get("trainer"))

        print(f"======= Training {self.cfg.get('model')['name']} =======")


        runner.fit(pl_model, datamodule=dm)
        torch.save(pl_model.state_dict(), os.path.join(
            '/home/skim/NAS01/Users/skim/model_save/self-supervised/simclr',
            'resnet50_50%DATASET.ckpt'))
