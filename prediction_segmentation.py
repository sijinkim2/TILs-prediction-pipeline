from pathlib import Path
import torch
from argparse import ArgumentParser
import argparse
import os
import numpy as np
from pytorch_lightning import LightningModule, Trainer, seed_everything
from collections import OrderedDict
from pl_bolts.models.vision.segmentation import SemSegment
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

tb_loggers = pl_loggers.TensorBoardLogger("logs/")
from torchvision import transforms
from torch.nn import functional as F
from torch import sigmoid, softmax
from PIL import Image
from torchvision.transforms import Lambda
from pathlib import Path
#from pl_bolts.models.vision.FROC import extract_predictions, non_max_supression_distance
import matplotlib.pyplot as plt

files = os.listdir("/home/skim/python_project/TIL_cropped_images")
save_path = os.path.join("/home/skim/python_project/TIL_segmentation_prediction/segmentation_ImageNet_fine_tuning")


def add_model_specific_args(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
    parser.add_argument("--num_layers", type=int, default=5, help="number of layers on u-net")
    parser.add_argument("--features_start", type=float, default=64, help="number of features in first layer")
    parser.add_argument(
        "--bilinear", action="store_true", default=False, help="whether to use bilinear interpolation or transposed"
    )

    return parser


if __name__ == "__main__":
    from pl_bolts.datamodules import KittiDataModule

    seed_everything(1234)

    parser = ArgumentParser()
    # trainer args
    parser = Trainer.add_argparse_args(parser)
    # model args
    parser = SemSegment.add_model_specific_args(parser)
    # datamodule args
    parser = KittiDataModule.add_argparse_args(parser)

    args = parser.parse_args()
    args.__dict__["gpus"] = 1
    args.__dict__["batch_size"] = 40
    args.__dict__["precision"] = 32
    args.__dict__["logger"] = tb_loggers
    args.__dict__["max_epochs"] = 100
    args.__dict__["callbacks"] = [ModelCheckpoint(save_top_k=2, save_last=True, monitor="val_loss"),
                                  # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
                                  LearningRateMonitor()]

    pretrained_path = f'/home/skim/model_save/Segmentation/Deeplabv3/ImageNet/resnet18/fold2_fine_tuning/checkpoints/epoch=44-step=17909.ckpt'

    dm = KittiDataModule(args.data_dir).from_argparse_args(args)

    model = SemSegment(**args.__dict__).load_from_checkpoint(pretrained_path)
    # model = SimCLR.load_from_checkpoint(pretrained_path, strict=False)
    model = model.to("cuda:0")
    model.eval()
    transform = transforms.ToTensor()
    a = 1

    for i in files:
        path = os.path.join('/home/skim/python_project/TIL_cropped_images', i)
        file_name = Path(path).stem
        os.mkdir(save_path + "/%s" % (file_name))
        files2 = os.listdir('/home/skim/python_project/TIL_cropped_images' + "/%s" % file_name)
        a = 1
        for j in files2:
            path2 = os.path.join('/home/skim/python_project/TIL_cropped_images' + "/%s" % file_name, j)
            file_name2 = Path(path2).stem
            img = Image.open(path2)
            dst = str(file_name2) + '.png'
            img = transform(img)
            img_array = np.array(img)
            img = torch.unsqueeze(img, dim=0)
            img = img.to('cuda:0')
            out = model.forward(img)

            out = F.softmax(out)
            out = out.detach().cpu().numpy()
            # np_pred = np.transpose(out, (0, 2, 3, 1))
            pred_array = np.array(out)
            argmax_pred = out.argmax(1)
            # out_array = np.array(out)
            print(a, file_name2)
            a = a + 1
            for i in range(len(argmax_pred)):
                argmax_pred0 = Lambda(lambda x: x[i, :, :])(argmax_pred)

                bmp = Image.fromarray(np.array(argmax_pred0, dtype='uint8'))

                # dst = str() +  +str(a) + '.png'
                pallette = [250,0,0, 0,250,0, 0,0,250]
                bmp.putpalette(pallette)
                bmp.save(save_path + "/%s/%s" % (file_name, dst))




