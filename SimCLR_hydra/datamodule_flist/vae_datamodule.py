from typing import Any, Callable, Optional

from pytorch_lightning import LightningDataModule

from torch.utils.data import DataLoader, Dataset
from datamodule_flist.cervix_dataset import CervixDataset

from torchvision import transforms as transform_lib

class VAEDataModule(LightningDataModule):
    """

    Cervical cell image train, val and test dataloaders.

     Example::

        from datamodules.cervix_datamodule import CervixDataModule

        dm = CervixDataModule(data_dir='./cytology_cervix/tile_imgs')
        dm.setup()
        show_img(dm.val_dataloader())

    """

    name = "cervix"

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

        self.train_list = cfg['train_list']
        self.val_list = cfg['val_list']

        self.image_size = cfg['image_size']
        self.image_ch = cfg['image_ch']
        self.shape = (self.image_ch, self.image_size, self.image_size)
        self.batch_size = cfg['batch_size']

        self.num_workers = cfg['num_workers']
        # self.num_class = num_class

        self.shuffle = cfg['shuffle']
        self.pin_memory = cfg['pin_memory']
        self.drop_last = cfg['drop_last']

    @property
    def num_classes(self) -> int:
        """
        Return:

            1000

        """
        return 2

    def setup(self):
        self.train_dataset = CervixDataset(self.train_list, transform=self.train_transform())
        self.val_dataset = CervixDataset(self.val_list, transform=self.val_transform())
        #self.test_dataset = CervixDataset(self.data_dir, transform = self.test_transform())

    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        """The train dataloader."""
        #transforms = self.train_transform()
        #self.train_dataset = ImageFolder(self.data_dir + '/train', transform=transforms)
        return self._data_loader(self.train_dataset, shuffle=self.shuffle)

    def val_dataloader(self, *args: Any, **kwargs: Any):
        """The val dataloader."""
        #transforms = self.val_transform()
        #self.val_dataset = ImageFolder(self.data_dir + '/val', transform=transforms)
        return self._data_loader(self.val_dataset)

    def test_dataloader(self, *args: Any, **kwargs: Any):
        """The val dataloader."""
        #transforms = self.val_transform()
        #self.val_dataset = ImageFolder(self.data_dir + '/val', transform=transforms)
        return self._data_loader(self.test_dataset)

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

    def train_transform(self) -> Callable:
        """The standard imagenet transforms.

        .. code-block:: python

            transform_lib.Compose([
                transform_lib.RandomResizedCrop(self.image_size),
                transform_lib.RandomHorizontalFlip(),
                transform_lib.ToTensor(),
                transform_lib.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        """
        preprocessing = transform_lib.Compose(
            [
                transform_lib.RandomResizedCrop(self.image_size),
                transform_lib.RandomHorizontalFlip(p=0.25),
                transform_lib.RandomVerticalFlip(p=0.25),
                #transform_lib.RandomRotation(degrees=(0, 180))
                # transform_lib.ColorJitter(brightness= 0,
                #                           contrast= 0,
                #                           saturation= 0.05,
                #                           hue= 0.05),
                transform_lib.ToTensor(),
                #agenet_normalization(),
            ]
        )

        return preprocessing

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

        preprocessing = transform_lib.Compose(
            [
                transform_lib.Resize(self.image_size),
                #transform_lib.CenterCrop(self.image_size),
                transform_lib.ToTensor(),
                #imagenet_normalization(),
            ]
        )

        return preprocessing

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
                #imagenet_normalization(),
            ]
        )

        return preprocessing




if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid, save_image
    import numpy as np
    from PIL import Image

    from datamodule_flist.vae_datamodule import VAEDataModule

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
        img = dataiter.next()
        # show images
        imshow(make_grid(img))
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


    parser = argparse.ArgumentParser(description='Generic runner for VAE models')
    parser.add_argument('--config', '-c',
                        dest="filename",
                        metavar='FILE',
                        help='path to the config file',
                        default='configs/vanillar.yaml')

    args = parser.parse_args()

    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)


    dm = VAEDataModule(config['datamodule'])

    #dm = VAEDataModule(train_list=train_list, val_list= val_list, image_size=256, batch_size=10, num_workers=8)
    dm.setup()
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    show_img(train_loader)
    #show_img_mask(test_loader)

    print('Test Done')