import os
from collections import OrderedDict
import torch.utils.data as data
from . import utils
import glob

class Woodscapes(data.Dataset):
    """Woodscapes dataset https://woodscape.valeo.com/.

    Keyword arguments:
    - root_dir (``string``): Root directory path.
    - mode (``string``): The type of dataset: 'train' for training set, 'val'
    for validation set, and 'test' for test set.
    - transform (``callable``, optional): A function/transform that  takes in
    an PIL image and returns a transformed version. Default: None.
    - label_transform (``callable``, optional): A function/transform that takes
    in the target and transforms it. Default: None.
    - loader (``callable``, optional): A function to load an image given its
    path. By default ``default_loader`` is used.

    """
    image_folder =  'rgb_images'
    lbl_folder = 'semantic_annotations/gtLabels'

    train_split = 'train.txt'
    val_split = 'val.txt'
    test_split = 'test.txt'

    # # Training dataset root folders
    # train_folder = "leftImg8bit_trainvaltest/leftImg8bit/train"
    # train_lbl_folder = "gtFine_trainvaltest/gtFine/train"

    # # Validation dataset root folders
    # val_folder = "leftImg8bit_trainvaltest/leftImg8bit/val"
    # val_lbl_folder = "gtFine_trainvaltest/gtFine/val"

    # # Test dataset root folders
    # test_folder = "leftImg8bit_trainvaltest/leftImg8bit/val"
    # test_lbl_folder = "gtFine_trainvaltest/gtFine/val"

    # Filters to find the images
    # img_extension = '.png'
    # lbl_name_filter = 'labelIds'

    # The values associated with the 35 classes
    full_classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    # The values above are remapped to the following
    new_classes = full_classes

    # Default encoding for pixel value, class name, and class color
    color_encoding = OrderedDict([
            ('unlabeled', (0, 0, 0)),
            ('road', (255, 0, 255)), # 6
            ('lanemarks', (255, 0, 0)),
            ('curb', (0, 255, 0)),
            ('person', (0, 0, 255)),
            ('rider', (255, 255, 255)),
            ('vehicles', (255, 255, 0)), # 0F
            ('bicycle', (0, 255, 255)),
            ('motorcycle', (128, 128, 255)),
            ('traffic_sign', (0, 128, 128))
    ])

    def __init__(self,
                 root_dir,
                 mode='train',
                 transform=None,
                 label_transform=None,
                 loader=utils.pil_loader):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.label_transform = label_transform
        self.loader = loader

        if self.mode.lower() == 'train':
            # Get the training data and labels filepaths
            split_path = os.path.join(root_dir, self.train_split)
            with open(split_path, "r") as f:
                file_list = f.read().splitlines()

            self.train_data = []
            self.train_labels = []
            for file in file_list:
                data_path = os.path.join(root_dir, self.image_folder, file)
                lbl_path = os.path.join(root_dir, self.lbl_folder, file)
                self.train_data += glob.glob(data_path)
                self.train_labels += glob.glob(lbl_path)

        elif self.mode.lower() == 'val':
            # Get the validation data and labels filepaths
            split_path = os.path.join(root_dir, self.val_split)
            with open(split_path, "r") as f:
                file_list = f.read().splitlines()

            self.val_data = []
            self.val_labels = []
            for file in file_list:
                data_path = os.path.join(root_dir, self.image_folder, file)
                lbl_path = os.path.join(root_dir, self.lbl_folder, file)
                self.val_data += glob.glob(data_path)
                self.val_labels += glob.glob(lbl_path)

        elif self.mode.lower() == 'test':
            # Get the test data and labels filepaths
            split_path = os.path.join(root_dir, self.test_split)
            with open(split_path, "r") as f:
                file_list = f.read().splitlines()

            self.test_data = []
            self.test_labels = []
            for file in file_list:
                data_path = os.path.join(root_dir, self.image_folder, file)
                lbl_path = os.path.join(root_dir, self.lbl_folder, file)
                self.test_data += glob.glob(data_path)
                self.test_labels += glob.glob(lbl_path)
        else:
            raise RuntimeError("Unexpected dataset mode. "
                               "Supported modes are: train, val and test")

    def __getitem__(self, index):
        """
        Args:
        - index (``int``): index of the item in the dataset

        Returns:
        A tuple of ``PIL.Image`` (image, label) where label is the ground-truth
        of the image.

        """
        if self.mode.lower() == 'train':
            data_path, label_path = self.train_data[index], self.train_labels[
                index]
        elif self.mode.lower() == 'val':
            data_path, label_path = self.val_data[index], self.val_labels[
                index]
        elif self.mode.lower() == 'test':
            data_path, label_path = self.test_data[index], self.test_labels[
                index]
        else:
            raise RuntimeError("Unexpected dataset mode. "
                               "Supported modes are: train, val and test")

        img, label = self.loader(data_path, label_path)

        # Greyscale
        label = label.convert('L')
        # label.convert('L').show()
        # exit()
        # Remap class labels

        label = utils.remap(label, self.full_classes, self.new_classes)

        if self.transform is not None:
            img = self.transform(img)


        if self.label_transform is not None:
            label = self.label_transform(label)
        # print(label.max())
        return img, label

    def __len__(self):
        """Returns the length of the dataset."""
        if self.mode.lower() == 'train':
            return len(self.train_data)
        elif self.mode.lower() == 'val':
            return len(self.val_data)
        elif self.mode.lower() == 'test':
            return len(self.test_data)
        else:
            raise RuntimeError("Unexpected dataset mode. "
                               "Supported modes are: train, val and test")
