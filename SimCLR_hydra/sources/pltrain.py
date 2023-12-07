import os
import torch
from torch import optim
import pytorch_lightning as pl

from pl_bolts.optimizers.lars import LARS
from pl_bolts.optimizers.lr_scheduler import linear_warmup_decay

class SSLearner(pl.LightningModule):
    def __init__(self, model, cfg):
        super().__init__()

        self.save_hyperparameters(cfg)
        self.model = model

        #self.trainer.datamodule.val_dataloader()
        # compute iters per epoch
        #global_batch_size = self.num_nodes * self.gpus * self.batch_size if self.gpus > 0 else self.batch_size
        #self.train_iters_per_epoch = self.num_samples // self.batch_size
        
        #self.current_device = torch.device("cuda")

    def shared_step(self, batch):

        img1, img2, _ = batch

        z1, z2 = self.model(img1, img2)

        loss = self.model.loss_function(z1, z2, self.hparams['temperature'])

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)

        self.log("train_loss", loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)

        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        # We will support Adam or SGD as optimizers.
        if self.hparams['optimizer'] == "Adam":
            # AdamW is Adam with a correct implementation of weight decay (see here
            # for details: https://arxiv.org/pdf/1711.05101.pdf)
            optimizer = optim.Adam(self.model.parameters(), lr = self.hparams['lr'], weight_decay=self.hparams['weight_decay'])

        elif self.hparams['optimizer'] == "Lars":
            optimizer = LARS(
                self.model.parameters(),
                lr= self.hparams['lr'],
                momentum=0.9,
                weight_decay=self.hparams['weight_decay'],
                trust_coefficient=0.001,
            )

        elif self.hparams['optimizer']  == "SGD":
            optimizer = optim.SGD(self.model.parameters(), lr = self.hparams['lr'], weight_decay=self.hparams['weight_decay'])
        else:
            assert False, f'Unknown optimizer: "{self.hparams.optimizer_name}"'

        ################# lr_scheduler ############################################
        if self.hparams['lr_scheduler'] == "MultiStep":
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 80], gamma=self.hparams['scheduler_gamma'])
        elif self.hparams['lr_scheduler'] == "Lambda":
            lambda1 = lambda epoch: 0.65 ** epoch
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
        elif self.hparams['lr_scheduler'] == "Cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0.0001)
        elif self.hparams['lr_scheduler'] == "Cyclic":
            scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr = 0.001, max_lr = self.hparams['lr'],
                                                    step_size_up=5, mode="exp_range", gamma=self.hparams['scheduler_gamma'])
        else:
            assert False, f'Unknown lr_scheduler: "{self.hparams.optimizer_name}"'

        return [optimizer], [scheduler]

