from argparse import ArgumentParser
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import torch
import torch.nn
from torchvision.utils import make_grid
from pytorch_lightning import LightningModule, Trainer, seed_everything
from torch.nn import functional as F
from pytorch_lightning import loggers as pl_loggers
from dice_loss import Class_wise_Dice_score
from torchmetrics.functional import dice_score

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from pl_bolts.models.vision.deeplabv3.model.deeplabv3 import DeepLabV3

tb_loggers = pl_loggers.TensorBoardLogger("logs/")


def one_hot_label(
        labels: torch.Tensor,
        num_classes: int = 3,
        device=torch.device('cuda:0'),
        dtype=torch.int32,
        eps: float = 1e-6,
        ignore_index=250,
) -> torch.Tensor:
    shape = labels.shape
    one_hot = torch.zeros((shape[0], ignore_index + 1) + shape[1:], device=device, dtype=dtype)
    one_hot = one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps
    ret = torch.split(one_hot, [num_classes, ignore_index + 1 - num_classes], dim=1)[0]

    return ret


class SemSegment(LightningModule):
    def __init__(
            self,
            lr: float = 0.01,
            num_classes: int = 3,
            num_layers: int = 5,
            features_start: int = 64,
            bilinear: bool = False,
            after_training: bool = True,
            **kwargs
    ):
        """Basic model for semantic segmentation. Uses UNet architecture by default.

        The default parameters in this model are for the KITTI dataset. Note, if you'd like to use this model as is,
        you will first need to download the KITTI dataset yourself. You can download the dataset `here.
        <http://www.cvlibs.net/datasets/kitti/eval_semseg.php?benchmark=semantics2015>`_

        Implemented by:

            - `Annika Brundyn <https://github.com/annikabrundyn>`_

        Args:
            num_layers: number of layers in each side of U-net (default 5)
            features_start: number of features in first layer (default 64)
            bilinear: whether to use bilinear interpolation (True) or transposed convolutions (default) for upsampling.
            lr: learning (default 0.01)
        """
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.features_start = features_start
        self.bilinear = bilinear
        self.lr = lr
        self.after_training = True
        self.net = DeepLabV3()

        print('fdfasd')

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_nb):
        img, mask = batch
        img = img.float()
        mask = mask.long()

        print(mask.max())

        label = one_hot_label(mask)

        out = self(img)
        pred = F.sigmoid(out)

        dice = dice_score(pred, mask)

        dice0, dice1, dice2, diceT, bce_loss, dice_loss, loss_val = Class_wise_Dice_score(out, label)

        self.log('train_dice2', dice, on_step=False, on_epoch=True)
        self.log('train_dice3', diceT, on_step=False, on_epoch=True)
        self.log('train_loss', loss_val, on_step=False, on_epoch=True)
        self.log('train_bce_loss', bce_loss, on_step=False, on_epoch=True)
        self.log('train_dice_loss', dice_loss, on_step=False, on_epoch=True)
        self.log('train_dice_other', dice0, on_step=False, on_epoch=True)
        self.log('train_dice_tumor', dice1, on_step=False, on_epoch=True)
        self.log('train_dice_stroma', dice2, on_step=False, on_epoch=True)

        return loss_val

    def validation_step(self, batch, batch_idx):
        img, mask = batch
        img = img.float()
        mask = mask.long()
        label = one_hot_label(mask)

        out = self(img)
        pred = F.softmax(out)
        np_mask = mask.detach().cpu().numpy()
        np_mask = np.transpose(np_mask, (0, 1, 2))
        np_pred = pred.detach().cpu().numpy()
        np_pred = np.transpose(np_pred, (0, 2, 3, 1))
        pred_array = np.array(np_pred)
        argmax_pred = np_pred.argmax(3)

        # Class_wise_dice_score
        dice0, dice1, dice2, diceT, bce_loss, dice_loss, loss_val = Class_wise_Dice_score(out, label)
        dice = dice_score(pred, mask)

        self.log('val_dice2', dice, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_dice3', diceT, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_loss', loss_val, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_bce_loss', bce_loss, on_step=False, on_epoch=True)
        self.log('val_dice_loss', dice_loss, on_step=False, on_epoch=True)
        self.log('val_dice_other', dice0, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_dice_tumor', dice1, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_dice_stroma', dice2, on_step=False, on_epoch=True, sync_dist=True)

        if (batch_idx == 0) and self.after_training:
            self.logger.experiment.add_image("val_input", make_grid(img, nrow=5))
            self.logger.experiment.add_image("val_label", make_grid(label[:, :3], nrow=5))
            self.logger.experiment.add_image("val_pred", make_grid(pred[:, :3], nrow=5))

        return loss_val

    def training_epoch_end(self, outputs):
        self.after_training = True
    '''
    def test_step(self, batch, batch_idx):
        test_img, test_mask = batch
        test_img = test_img.float()
        test_mask = test_mask.long()

        label = one_hot_label(test_mask.to(torch.int64))

        out = self(test_img)
        pred = F.sigmoid(out)
        np_mask = test_mask.detach().cpu().numpy()
        np_mask = np.transpose(np_mask, (0, 1, 2))
        np_pred = pred.detach().cpu().numpy()
        np_pred = np.transpose(np_pred, (0, 2, 3, 1))
        pred_array = np.array(np_pred)
        argmax_pred = np_pred.argmax(3)

        a = 1

        # if (batch_idx == 0) and self.after_training:
        # for i in range(len(argmax_pred)):
        # argmax_pred0 = Lambda(lambda x: x[i, :, :])(argmax_pred)

        # bmp = Image.fromarray(np.array(argmax_pred0, dtype='uint8'))
        # dst = 'pred_image' + str(a) + '.png'
        # pallette = [250,0,0 , 0,250,0 , 0,0,250 , 250,250,0, 250,250,250, 250,0,250, 0,250,250, 0,0,0]
        # pallette = [250,0,0, 0,250,0, 0,0,250]
        # bmp.putpalette(pallette)
        # a = a + 1
        # bmp.save(
        # '/home/skim/PycharmProjects/pythonProject/lightning-bolts-master/pl_bolts/models/vision/test_pred/%s' % dst)

        # Dice_score
        # dice_loss = DiceBCELoss()
        # loss_val, val_dice = dice_loss(out, label)
        dice = dice_score(pred, test_mask)
        # diceloss = DiceLoss()
        # dice_loss = diceloss(out, label)
        # bce = F.binary_cross_entropy_with_logits(pred, label)
        # loss_val = dice_loss + bce
        # Class_wise_dice_score
        dice0, dice1, dice2, diceT, bce_loss, dice_loss, loss_val = Class_wise_Dice_score(out, label)

        # self.log('test_dice1', val_dice, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test_dice2', dice, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test_dice3', diceT, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test_loss', loss_val, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test_bce_loss', bce_loss, on_step=False, on_epoch=True, sync_dist=True)
        #self.log('test_dice_loss', dice_val, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test_dice_other', dice0, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test_dice_tumor', dice1, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test_dice_stroma', dice2, on_step=False, on_epoch=True, sync_dist=True)
    '''
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)
        return [opt], [sch]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
        parser.add_argument("--num_layers", type=int, default=5, help="number of layers on u-net")
        parser.add_argument("--features_start", type=float, default=64, help="number of features in first layer")
        parser.add_argument(
            "--bilinear", action="store_true", default=False, help="whether to use bilinear interpolation or transposed"
        )

        return parser


def cli_main():
    from TIGER_datamodule import TIGERDataModule

    seed_everything(1234)

    parser = ArgumentParser()
    # trainer args
    parser = Trainer.add_argparse_args(parser)
    # model args
    parser = SemSegment.add_model_specific_args(parser)
    # datamodule args
    parser = TIGERDataModule.add_argparse_args(parser)

    args = parser.parse_args()
    args.__dict__["gpus"] = 1
    args.__dict__["batch_size"] = 40
    args.__dict__["precision"] = 32
    args.__dict__["logger"] = tb_loggers
    args.__dict__["max_epochs"] = 100
    args.__dict__["num_workers"] = 20
    args.__dict__["callbacks"] = [ModelCheckpoint(save_top_k=2, save_last=True, monitor="val_loss"),
                                  # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
                                  LearningRateMonitor()]  # , EarlyStopping(monitor="val_loss", mode="min", patience=5 )]

    # data
    dm = TIGERDataModule(args.data_dir).from_argparse_args(args)

    # model
    model = SemSegment(**args.__dict__)

    # train
    trainer = Trainer().from_argparse_args(args)
    trainer.fit(model, datamodule=dm)
    model.eval()
    #trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    cli_main()
