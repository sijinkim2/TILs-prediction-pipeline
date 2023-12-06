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
from pl_bolts.models.vision.FROC import extract_predictions, non_max_supression_distance
import matplotlib.pyplot as plt


def one_hot_label_detection(
        labels: torch.Tensor,
        num_classes: int = 2,
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

def one_hot_label_segmentation(
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


files = os.listdir("path")

for i in files:
    seg_path = os.path.join('path', i)
    det_path = os.path.join('path', i)

    file_name = Path(seg_path).stem
    files2 = os.listdir('path' + "/%s" % file_name)

    Num_of_Stroma = 0
    Num_of_Lymphocytes = 0
    TIL = 0
    for j in files2:
        seg_path2 = os.path.join('path' + "/%s" % file_name, j)
        det_path2 = os.path.join('path' + "/%s" % file_name, j)


        trasform = transforms.ToTensor()
        seg_map = Image.open(seg_path2)
        det_map = Image.open(det_path2)

        seg_map = np.array(seg_map)
        det_map = np.array(det_map)
        det_map = np.reshape(det_map,(1,) + det_map.shape)

        predicted_detections = []

        for i in range(len(det_map)):
            #print(i)
            temp1 = extract_predictions(det_map[i], confidence_threshold=0.1)
            predicted_detections.append(non_max_supression_distance(temp1, distance_threshold=12))

        del temp1

        if np.count_nonzero(det_map == 1) > 0:
            a = 0
            for i in range(len(predicted_detections[0])):

                x = predicted_detections[0][a][0]
                y = predicted_detections[0][a][1]
                a = a + 1

                #print(seg_map[x, y])
                #print(file_name)
                if seg_map[x, y] == 2:
                    TIL = TIL + 1
        Num_of_Stroma = Num_of_Stroma + np.count_nonzero(seg_map == 2)
        Num_of_Lymphocytes = Num_of_Lymphocytes + a
        #print(Num_of_Stroma)

    TIL_score = 100 * ((TIL * 16 * 16) / Num_of_Stroma)

    print(file_name, Num_of_Lymphocytes, TIL, Num_of_Stroma, round(TIL_score))

