from pathlib import Path
import torch
from argparse import ArgumentParser
import argparse
import os
import numpy as np
from pytorch_lightning import LightningModule, Trainer, seed_everything
from collections import OrderedDict
from pl_bolts.models.vision.detection import SemSegment
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
tb_loggers = pl_loggers.TensorBoardLogger("logs/")
from torchvision import transforms
from torch.nn import functional as F
from torch import sigmoid, softmax
from PIL import Image
from torchvision.transforms import Lambda
from pathlib import Path
from pl_bolts.models.vision.dice_loss import extract_predictions, non_max_supression_distance
import matplotlib.pyplot as plt

files = os.listdir("image_path")
save_path = os.path.join("save_path")



    pretrained_path = f'resnet18.ckpt'

    model = SemSegment(**args.__dict__).load_from_checkpoint(pretrained_path)
    model = model.to("cuda:0")
    model.eval()
    transform = transforms.ToTensor()
    a = 1

    for i in files:
        path = os.path.join('image_path', i)
        file_name = Path(path).stem
        os.mkdir(save_path + "/%s" % (file_name))
        files2 = os.listdir('image_path' + "/%s" % file_name)
        a = 1
        for j in files2:
            path2 = os.path.join('image_path' + "/%s" % file_name, j)
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
                pallette = [250,0,0, 0,250,0]
                bmp.putpalette(pallette)
                bmp.save(save_path + "/%s/%s" % (file_name, dst))




