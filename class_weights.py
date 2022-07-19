## Script for calculating class weights
# Supports 2 weighting method
# - enet_weighting
# - median_frequency weighting
# Author: Jinsu ha , soo4826@gmail.com
# Date: 2022.5.30

import numpy as np
from PIL import Image
from torch import median
from data import Woodscapes as dataset
from tqdm import tqdm

## Script for calculating class weights
# Supports 2 weighting method
# - enet_weighting
# - median_frequency weighting


def enet_weighting(dataloader, num_classes, c=1.02):
    """Computes class weights as described in the ENet paper:

        w_class = 1 / (ln(c + p_class)),

    where c is usually 1.02 and p_class is the propensity score of that
    class:

        propensity_score = freq_class / total_pixels.

    References: https://arxiv.org/abs/1606.02147

    Keyword arguments:
    - dataloader (``data.Dataloader``): A data loader to iterate over the
    dataset.
    - num_classes (``int``): The number of classes.
    - c (``int``, optional): AN additional hyper-parameter which restricts
    the interval of values for the weights. Default: 1.02.

    """
    class_count = 0
    total = 0
    for _, label in tqdm(dataloader):
        # label = label.cpu().numpy()
        label = np.array(label)

        # Flatten label
        flat_label = label.flatten()

        # Sum up the number of pixels of each class and the total pixel
        # counts for each label
        class_count += np.bincount(flat_label, minlength=num_classes)
        total += flat_label.size

    # Compute propensity score and then the weights for each class
    propensity_score = class_count / total
    class_weights = 1 / (np.log(c + propensity_score))

    return class_weights


def median_freq_balancing(dataloader, num_classes):
    """Computes class weights using median frequency balancing as described
    in https://arxiv.org/abs/1411.4734:

        w_class = median_freq / freq_class,

    where freq_class is the number of pixels of a given class divided by
    the total number of pixels in images where that class is present, and
    median_freq is the median of freq_class.

    Keyword arguments:
    - dataloader (``data.Dataloader``): A data loader to iterate over the
    dataset.
    whose weights are going to be computed.
    - num_classes (``int``): The number of classes

    """
    class_count = 0
    total = 0
    for _, label in tqdm(dataloader):
        # label = label.cpu().numpy()
        label = np.array(label)

        # Flatten label
        flat_label = label.flatten()

        # Sum up the class frequencies
        bincount = np.bincount(flat_label, minlength=num_classes)

        # Create of mask of classes that exist in the label
        mask = bincount > 0
        # Multiply the mask by the pixel count. The resulting array has
        # one element for each class. The value is either 0 (if the class
        # does not exist in the label) or equal to the pixel count (if
        # the class exists in the label)
        total += mask * flat_label.size

        # Sum up the number of pixels found for each class
        class_count += bincount

    # Compute the frequency and its median
    freq = class_count / total
    med = np.median(freq)

    return med / freq




if __name__ == "__main__":
    root_dir = "dataset/Woodscape"
    dataset_train = dataset(root_dir, 'train')
    median_freq_weight = median_freq_balancing(dataset_train, 10)
    enet_weight = enet_weighting(dataset_train, 10)

    print("Median freq weight: ", median_freq_weight)
    print("Enet weight: ", enet_weight)
