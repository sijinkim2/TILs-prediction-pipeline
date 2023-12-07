import os

import numpy as np
from torch.utils.data import Dataset

from pl_bolts.utils import _PIL_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg

if _PIL_AVAILABLE:
    from PIL import Image
else:  # pragma: no cover
    warn_missing_pkg("PIL", pypi_name="Pillow")

#tiger dataset
DEFAULT_VALID_LABELS = (0, 1 ,2)
#DEFAULT_VALID_LABELS = (0, 1)

class Train_Dataset(Dataset):
    """
    Note:
        You need to have downloaded the Kitti dataset first and provide the path to where it is saved.
        You can download the dataset here: http://www.cvlibs.net/datasets/kitti/eval_semseg.php?benchmark=semantics2015

    There are 34 classes, however not all of them are useful for training (e.g. railings on highways). These
    useless classes (the pixel values of these classes) are stored in `void_labels`. Useful classes are stored
    in `valid_labels`.

    The `encode_segmap` function sets all pixels with any of the `void_labels` to `ignore_index`
    (250 by default). It also sets all of the valid pixels to the appropriate value between 0 and
    `len(valid_labels)` (since that is the number of valid classes), so it can be used properly by
    the loss function when comparing with the output.
    """

    IMAGE_PATH = os.path.join("IMAGE_PATH")
    MASK_PATH = os.path.join("MASK_PATH")
    def __init__(
        self,
        data_dir: str,
        img_size: tuple = (256, 256),
        #void_labels: list = DEFAULT_VOID_LABELS,
        valid_labels: list = DEFAULT_VALID_LABELS,
        transform=None,
    ):
        """
        Args:
            data_dir (str): where to load the data from path, i.e. '/path/to/folder/with/data_semantics/'
            img_size: image dimensions (width, height)
            void_labels: useless classes to be excluded from training
            valid_labels: useful classes to include
        """
        if not _PIL_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError("You want to use `PIL` which is not installed yet.")

        self.img_size = img_size
        #self.void_labels = void_labels
        self.valid_labels = valid_labels
        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_labels, range(len(self.valid_labels))))
        self.transform = transform

        self.data_dir = data_dir
        self.img_path = os.path.join(self.data_dir, self.IMAGE_PATH) #train img path
        self.mask_path = os.path.join(self.data_dir, self.MASK_PATH) # train mask path
        self.img_list = self.get_filenames(self.img_path)
        self.mask_list = self.get_filenames(self.mask_path)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx])
        img = img.resize(self.img_size)
        img = np.array(img)


        mask = Image.open(self.mask_list[idx]).convert("L")
        mask = mask.resize(self.img_size)
        mask = np.array(mask)
        mask = self.encode_segmap(mask)

        if self.transform:
            img = self.transform(img)



        return img, mask


    def get_filenames(self, path):
        """Returns a list of absolute paths to images inside given `path`"""
        files_list = list()
        for filename in os.listdir(path):
            files_list.append(os.path.join(path, filename))
        return files_list


    def encode_segmap(self, mask):
        """Sets void classes to zero so they won't be considered for training."""
        #for voidc in self.void_labels:
        #    mask[mask == voidc] = self.ignore_index
        for validc in self.valid_labels:
            mask[mask == validc] = self.class_map[validc]
        # remove extra idxs from updated dataset
        #mask[mask > 18] = self.ignore_index
        return mask



class val_Dataset(Dataset):
    """
    Note:
        You need to have downloaded the Kitti dataset first and provide the path to where it is saved.
        You can download the dataset here: http://www.cvlibs.net/datasets/kitti/eval_semseg.php?benchmark=semantics2015

    There are 34 classes, however not all of them are useful for training (e.g. railings on highways). These
    useless classes (the pixel values of these classes) are stored in `void_labels`. Useful classes are stored
    in `valid_labels`.

    The `encode_segmap` function sets all pixels with any of the `void_labels` to `ignore_index`
    (250 by default). It also sets all of the valid pixels to the appropriate value between 0 and
    `len(valid_labels)` (since that is the number of valid classes), so it can be used properly by
    the loss function when comparing with the output.
    """

    IMAGE_PATH = os.path.join(
        "IMAGE_PATH")
    MASK_PATH = os.path.join(
        "MASK_PATH")

    def __init__(
            self,
            data_dir: str,
            img_size: tuple = (256, 256),
            # void_labels: list = DEFAULT_VOID_LABELS,
            valid_labels: list = DEFAULT_VALID_LABELS,
            transform=None,
    ):
        """
        Args:
            data_dir (str): where to load the data from path, i.e. '/path/to/folder/with/data_semantics/'
            img_size: image dimensions (width, height)
            void_labels: useless classes to be excluded from training
            valid_labels: useful classes to include
        """
        if not _PIL_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError("You want to use `PIL` which is not installed yet.")

        self.img_size = img_size
        # self.void_labels = void_labels
        self.valid_labels = valid_labels
        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_labels, range(len(self.valid_labels))))
        self.transform = transform

        self.data_dir = data_dir
        self.img_path = os.path.join(self.data_dir, self.IMAGE_PATH)  # train img path
        self.mask_path = os.path.join(self.data_dir, self.MASK_PATH)  # train mask path
        self.img_list = self.get_filenames(self.img_path)
        self.mask_list = self.get_filenames(self.mask_path)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx])
        img = img.resize(self.img_size)
        img = np.array(img)

        mask = Image.open(self.mask_list[idx]).convert("L")
        mask = mask.resize(self.img_size)
        mask = np.array(mask)
        mask = self.encode_segmap(mask)

        if self.transform:
            img = self.transform(img)

        return img, mask

    def get_filenames(self, path):
        """Returns a list of absolute paths to images inside given `path`"""
        files_list = list()
        for filename in os.listdir(path):
            files_list.append(os.path.join(path, filename))
        return files_list

    def encode_segmap(self, mask):
        """Sets void classes to zero so they won't be considered for training."""
        # for voidc in self.void_labels:
        #    mask[mask == voidc] = self.ignore_index
        for validc in self.valid_labels:
            mask[mask == validc] = self.class_map[validc]
        # remove extra idxs from updated dataset
        # mask[mask > 18] = self.ignore_index
        return mask

'''
# test dataset
class test_Dataset(Dataset):
        """
        Note:
            You need to have downloaded the Kitti dataset first and provide the path to where it is saved.
            You can download the dataset here: http://www.cvlibs.net/datasets/kitti/eval_semseg.php?benchmark=semantics2015

        There are 34 classes, however not all of them are useful for training (e.g. railings on highways). These
        useless classes (the pixel values of these classes) are stored in `void_labels`. Useful classes are stored
        in `valid_labels`.

        The `encode_segmap` function sets all pixels with any of the `void_labels` to `ignore_index`
        (250 by default). It also sets all of the valid pixels to the appropriate value between 0 and
        `len(valid_labels)` (since that is the number of valid classes), so it can be used properly by
        the loss function when comparing with the output.
        """

        IMAGE_PATH = os.path.join(
            "IMAGE_PATH")
        MASK_PATH = os.path.join(
            "MASK_PATH")

        def __init__(
                self,
                data_dir: str,
                img_size: tuple = (512, 512),
                # void_labels: list = DEFAULT_VOID_LABELS,
                valid_labels: list = DEFAULT_VALID_LABELS,
                transform=None,
        ):
            """
            Args:
                data_dir (str): where to load the data from path, i.e. '/path/to/folder/with/data_semantics/'
                img_size: image dimensions (width, height)
                void_labels: useless classes to be excluded from training
                valid_labels: useful classes to include
            """
            if not _PIL_AVAILABLE:  # pragma: no cover
                raise ModuleNotFoundError("You want to use `PIL` which is not installed yet.")

            self.img_size = img_size
            # self.void_labels = void_labels
            self.valid_labels = valid_labels
            self.ignore_index = 250
            self.class_map = dict(zip(self.valid_labels, range(len(self.valid_labels))))
            self.transform = transform

            self.data_dir = data_dir
            self.img_path = os.path.join(self.data_dir, self.IMAGE_PATH)  # train img path
            self.mask_path = os.path.join(self.data_dir, self.MASK_PATH)  # train mask path
            self.img_list = self.get_filenames(self.img_path)
            self.mask_list = self.get_filenames(self.mask_path)

        def __len__(self):
            return len(self.img_list)

        def __getitem__(self, idx):
            img = Image.open(self.img_list[idx])
            img = img.resize(self.img_size)
            img = np.array(img)

            mask = Image.open(self.mask_list[idx]).convert("L")
            mask = mask.resize(self.img_size)
            mask = np.array(mask)
            mask = self.encode_segmap(mask)

            if self.transform:
                img = self.transform(img)

            return img, mask

        def get_filenames(self, path):
            """Returns a list of absolute paths to images inside given `path`"""
            files_list = list()
            for filename in os.listdir(path):
                files_list.append(os.path.join(path, filename))
            return files_list

        def encode_segmap(self, mask):
            """Sets void classes to zero so they won't be considered for training."""
            # for voidc in self.void_labels:
            #    mask[mask == voidc] = self.ignore_index
            for validc in self.valid_labels:
                mask[mask == validc] = self.class_map[validc]
            # remove extra idxs from updated dataset
            # mask[mask > 18] = self.ignore_index
            return mask
'''
