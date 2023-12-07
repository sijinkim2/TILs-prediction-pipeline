from typing import Any, Callable, Optional
import torch
import os
from pytorch_lightning import LightningDataModule
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader, Dataset
from pl_bolts.models.self_supervised.swav.pathology_dataset import PathologyDataset
from pl_bolts.models.self_supervised.simclr.transforms import SimCLRTrainDataTransform, SimCLREvalDataTransform
from pl_bolts.models.self_supervised.swav.transforms import SwAVTrainDataTransform, SwAVEvalDataTransform


from torchvision import transforms as transform_lib


file_path = "/home/skim/practice"

with open(file_path) as f:
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


class PathologyDataModule(LightningDataModule):
    """

    Cervical cell image train, val and test dataloaders.

     Example::

        from datamodules.cervix_datamodule import CervixDataModule

        dm = CervixDataModule(data_dir='./cytology_cervix/tile_imgs')
        dm.setup()
        show_img(dm.val_dataloader())

    """

    name = "pathology"

    def __init__(
            self,
            val_split: float = 0.2,
            test_split: float = 0.0,
            data_sample=0,
            data_zero=0,
            num_class: int = 1,
            image_size: int = 256,
            jitter_strength: float = 0.2,
            gaussian_blur: bool = True,
            num_workers: int = 8,
            batch_size: int = 128,
            shuffle: bool = True,
            pin_memory: bool = True,
            drop_last: bool = True,
            da_type: str = 'none',
            *args: Any,
            **kwargs: Any,
    ) -> None:
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
        super().__init__(*args, **kwargs)

        # if not _TORCHVISION_AVAILABLE:  # pragma: no cover
        #     raise ModuleNotFoundError(
        #         "You want to use ImageNet dataset loaded from `torchvision` which is not installed yet."
        #     )

        self.image_size = image_size
        self.jitter_strength = jitter_strength
        self.gaussian_blur = gaussian_blur
        self.shape = (3, self.image_size, self.image_size)

        self.train_list = all_X_list
        # self.val_list = val_list

        self.num_workers = num_workers
        # self.meta_dir = meta_dir
        self.num_class = num_class
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.da_type = da_type
        # self.num_samples = 1281167 - self.num_imgs_per_val_class * self.num_classes

        unused_len = round(0 * len(all_X_list))
        zero_len = round(0 * len(all_X_list))
        used_len = (len(all_X_list) - unused_len - zero_len)

        self.used_set, self.unused_set, self.zero_set = random_split(
            all_X_list, lengths=[used_len, unused_len, zero_len], generator=torch.Generator().manual_seed(42))

        val_len = round(val_split * len(self.used_set))
        test_len = round(test_split * len(self.used_set))
        train_len = len(self.used_set) - val_len - test_len

        self.trainset, self.valset, self.testset = random_split(
            self.used_set, lengths=[train_len, val_len, test_len], generator=torch.Generator().manual_seed(42)
        )

        self.train_dataset = PathologyDataset((self.trainset),
                                              transform=self.simclr_train_transform())
        self.val_dataset = PathologyDataset((self.valset), transform=self.simclr_val_transform())


    @property
    def num_classes(self) -> int:
        """
        Return:

            1000

        """
        return 10

    @property
    def num_samples(self) -> int:
        return len(self.train_list)

    '''
    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        """The train dataloader."""
        #transforms = self.train_transform()
        #self.train_dataset = ImageFolder(self.data_dir + '/train', transform=transforms)
        return self._data_loader(self.train_dataset, shuffle=self.shuffle)
    '''

    def train_dataloader(self) -> DataLoader:
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,

        )
        return loader

    # def val_dataloader(self, *args: Any, **kwargs: Any):
    #     dataloader1 = self._data_loader(self.val_dataset)
    #     dataloader2 = self._data_loader(self.cls_dataset)
    #     dataloader3 = self._data_loader(self.test_dataset)
    #     loaders = {"val_loader": dataloader1, "cls_loader": dataloader2, "test_loader": dataloader3}
    #     combined = CombinedLoader(loaders, mode="max_size_cycle")
    #     return combined

    def val_dataloader(self) -> DataLoader:
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
        return loader

    '''
    def val_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        return self._data_loader(self.val_dataset, shuffle=self.shuffle)
    '''

    def test_dataloader(self, *args: Any, **kwargs: Any):
        """The val dataloader."""
        # transforms = self.val_transform()
        # self.val_dataset = ImageFolder(self.data_dir + '/val', transform=transforms)
        return self._data_loader(self.testset)

    def _data_loader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )

    def swav_train_transform(self) -> Callable:
        return SwAVTrainDataTransform()

    def simclr_train_transform(self) -> Callable:
        return SimCLRTrainDataTransform()

    def simclr_val_transform(self) -> Callable:
        return SimCLREvalDataTransform()

    def swav_val_transform(self) -> Callable:
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

        # preprocessing = transform_lib.Compose(
        #     [
        #         transform_lib.Resize(self.image_size),
        #         transform_lib.CenterCrop(self.image_size),
        #         transform_lib.ToTensor(),
        #         #imagenet_normalization(),
        #     ]
        # )

        return SwAVEvalDataTransform()


    def test_transform(self) -> Callable:
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

        preprocessing = transform_lib.Compose(
            [
                transform_lib.Resize(self.image_size),
                transform_lib.CenterCrop(self.image_size),
                transform_lib.ToTensor(),
                # imagenet_normalization(),
            ]
        )

        return preprocessing

    # def test_transform(self) -> Callable:
    #     """The standard imagenet transforms for validation.
    #
    #     .. code-block:: python
    #
    #         transform_lib.Compose([
    #             transform_lib.Resize(self.image_size + 32),
    #             transform_lib.CenterCrop(self.image_size),
    #             transform_lib.ToTensor(),
    #             transform_lib.Normalize(
    #                 mean=[0.485, 0.456, 0.406],
    #                 std=[0.229, 0.224, 0.225]
    #             ),
    #         ])
    #     """
    #
    #     preprocessing = transform_lib.Compose(
    #         [
    #             transform_lib.Resize(self.image_size),
    #             transform_lib.CenterCrop(self.image_size),
    #             transform_lib.ToTensor(),
    #             #imagenet_normalization(),
    #         ]
    #     )
    #
    #     return SimCLREvalDataTransform


'''
if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid, save_image
    import numpy as np
    from utils import yaml_config_hook
    import argparse
    from PIL import Image

    from Train_Cervix_Datamodule import Train_DataModule

    def imshow(img):
        # img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    def show_img(dataloader):

        # get some random training images
        dataiter = iter(dataloader)
        img1, img2 = dataiter.next()
        # show images
        imshow(make_grid(img1))
        # save_image(make_grid(xi), 'SimCLR1_xi.png')


    def show_img_mask(dataloader):
        # get some random training images
        dataiter = iter(dataloader)
        img, mask = dataiter.next()
        # show images
        print(img.shape)
        print(mask.shape)
        imshow(make_grid(img))

        # save_image(make_grid(xi), 'SimCLR1_xi.png')

    parser = argparse.ArgumentParser(description="SimCLR")
    config = yaml_config_hook('/home/jchun/PycharmProject/drbrd/config/configs_SimCLR')
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #data_dir = '/mnt/jcho/Databank/2_Pathology/in_house/cytology_cervix/tile_class_all_abnormal'
    train_list = '/home/skim/PycharmProjects/pythonProject/lightning-bolts-master/pl_bolts/models/self_supervised/moco/pathology_dataset_list'
    val_list =  '/home/skim/PycharmProjects/pythonProject/lightning-bolts-master/pl_bolts/models/self_supervised/moco/practice'

    dm = Train_DataModule(train_list=args.train_list, val_list=args.val_list, batch_size=args.batch_size,
        image_size=args.image_size, num_workers=args.num_workers, da_type=args.da_type,
        jitter_strength=args.jitter_strength, gaussian_blur=args.gaussian_blur)
    dm.setup()

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    show_img(train_loader)
    #show_img_mask(test_loader)

    print('Test Done')
'''
