from typing import Any, Callable, Optional

import torchvision.transforms
from pytorch_lightning import LightningDataModule
from torchvision.datasets import ImageFolder
import os
from torch.utils.data.dataset import random_split
import torch
from torch.utils.data import DataLoader, Dataset
from pl_bolts.datasets.pathology_dataset import PathologyDataset
from torchvision import transforms

from pl_bolts.models.self_supervised.simclr.transforms import SimCLRTrainDataTransform, SimCLREvalDataTransform



class SimCLRDataModule(LightningDataModule):

    name = "pathology"

    def __init__(self, cfg):
        """
        Args:
            data_dir: path to the imagenet dataset file
            meta_dir: path to meta.bin file
            num_imgs_per_val_class: how many images per class for the validation set
            image_size: final image size
            num_workers: how many data workers
            batch_size: batch_size
            shuffle: If true shuffles the data every epoch
            pin_memory: If true, the data loader will copy Tensors into CUDA pinned memory before
                        returning them
            drop_last: If true drops the last incomplete batch
        """
        super().__init__()

        self.val_split = cfg['val_split']
        self.test_split = cfg['test_split']
        self.data_sample = 0
        self.data_zero = 0
        self.data_dir = cfg['data_dir']
        self.image_size = cfg['image_size']
        self.image_ch = cfg['image_ch']
        self.shape = (self.image_ch, self.image_size, self.image_size)
        self.batch_size = cfg['batch_size']
        self.seed = cfg['seed']

        self.num_workers = cfg['num_workers']
        #self.num_class = num_class

        self.shuffle = cfg['shuffle']
        self.pin_memory = cfg['pin_memory']
        self.drop_last = cfg['drop_last']

        self.gaussian_blur = cfg['gaussian_blur']
        self.jitter_strength = cfg['jitter_strength']

        self.file_path = cfg['file_path']
        
        with open(self.file_path) as f:
            lines = f.readlines()

        lines = [line.rstrip('\n') for line in lines]

        all_X_list = []

        for line in lines:
            l_folder_path = line.split()  # folder path
            img_folder_path = os.path.join(l_folder_path[0])  # img folder path

            file_lists = os.listdir(img_folder_path)  # images list in folder

            for img_name in file_lists:
                img_path = os.path.join(img_folder_path, img_name)
                all_X_list.append(img_path)
                # img = Image.open(img_path)
                # img.show()
                # print(all_X_list)

        f.close()
	
        unused_len = round(self.data_sample * len(all_X_list))
        zero_len = round(self.data_zero * len(all_X_list))
        used_len = (len(all_X_list) - unused_len - zero_len)

        self.used_set, self.unused_set, self.zero_set = random_split(
            all_X_list, lengths=[used_len, unused_len, zero_len], generator=torch.Generator().manual_seed(self.seed)
        )

        self.img_list = all_X_list
        


        val_len = round(self.val_split * len(self.img_list))
        test_len = round(self.test_split * len(self.img_list))
        train_len = (len(self.img_list) - val_len - test_len)

        self.trainset, self.valset, self.testset = random_split(
            self.img_list, lengths=[train_len, val_len, test_len], generator=torch.Generator().manual_seed(self.seed)
        )
        #self.num_samples = 1281167 - self.num_imgs_per_val_class * self.num_classes

        self.train_dataset = PathologyDataset(self.trainset, transform=SimCLRTrainDataTransform())

        self.val_dataset = PathologyDataset(self.valset, transform=SimCLREvalDataTransform())

    @property
    def num_classes(self) -> int:
        """
        Return:

            1000

        """
        return 1

    '''
    def setup(self):
        
        
        self.train_dataset = ImageFolder(self.data_dir + '/train',
                                         transform=SimCLRTrainDataTransform(self.image_size, self.gaussian_blur,
                                                                            self.jitter_strength))
        self.val_dataset = ImageFolder(self.data_dir + '/val',
                                       transform=SimCLREvalDataTransform(self.image_size, self.gaussian_blur,
                                                                          self.jitter_strength))
    '''
    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        """The train dataloader."""

        return self._data_loader(self.train_dataset, shuffle=self.shuffle)

    def val_dataloader(self, *args: Any, **kwargs: Any):
        """The val dataloader."""

        return self._data_loader(self.val_dataset)

    def train_transform(self) -> Callable:

        return SimCLRTrainDataTransform

    def val_transform(self) -> Callable:

        return SimCLREvalDataTransform

    # def test_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
    #     """The test dataloader."""
    #     return self._data_loader(self.dataset_test)

    def _data_loader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle = shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )

    def val_transform(self) -> Callable:
        """The standard imagenet transforms for validation.

        .. code-block:: python

            transform_lib.Compose([
                transform_lib.Resize(self.image_size + 32),
                transform_lib.CenterCrop(self.image_size),
                transform_lib.ToTensor(),
                transform_lib.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        """

        preprocessing = transforms.Compose(
            [
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                #imagenet_normalization(),
            ]
        )

        return preprocessing



'''
if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid
    import numpy as np

    from datamodule.simclr_datamodule import SimCLRDataModule

    import yaml
    import argparse

    def imshow(img):
        # img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    def show_img(dataloader):

        # get some random training images
        dataiter = iter(dataloader)

        (img1, img2, _), y = dataiter.next()
        #img1, y = dataiter.next()
        # show images
        imshow(make_grid(img2))
        # save_image(make_grid(xi), 'SimCLR1_xi.png')

    parser = argparse.ArgumentParser(description='Generic runner for VAE models')
    parser.add_argument('--config', '-c',
                        dest="filename",
                        metavar='FILE',
                        help='path to the config file',
                        default='configs/simCLR.yaml')

    args = parser.parse_args()

    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    dm = SimCLRDataModule(config['datamodule'])

    dm.setup()
    show_img(dm.train_dataloader())
   
    '''
