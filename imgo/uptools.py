"""
IMGO - Process, augment, and balance image data.
------------------------------------------------
UPTOOLS module: 

Last updated: version 2.5.4

Classes
-------
Image_Dataset: Class representing an image dataset, being a collection
of X (square image data) and y (label data) arrays.

    Class Attributes:
        base_path (str): path to the directory containing images or class
        subdirectories.
        -
        mode (str): format of source image data: "img" if raw images, 
        "np" if numpy-arrays, or "h5" if HDF5 format.
        -
        reduce (str): statistical reduction performed on raw data: 
        "norm" for pixel value normalization, "std" for pixel value 
        standardization (using training data statistics). Note that 
        any reduction will be undone if the datasets are saved to disk.
        -
        class_list (list): list of classes in the dataset.
        -
        class_no (int): number of classes in the dataset.
        -
        split (int): number of splits in the dataset: 2 if split into 
        training, validation, and testing subsets; 1 if split into 
        training and testing subsets; and 0 if not split (or merged).
        -
        dims (tuple): dimensions of the images in the dataset (set to 
        "various" (str) if the images are not of a consistent size).
        -
        img_shape (tuple): dimensions and channels of the images in the
        dataset.
        -
        expand (str): statistical expansion performed on raw data:
        "de_norm" for de-normalization of pixel values, and "de_std" for 
        de-standardization of pixel values.
        -
        shadow (dict): image and label data for each data subset "train", 
        "val", "test", and "data" (if unsplit or merged), in integer 
        (ie non-normalized and non-standardized form).
        -
        X_train (numpy-array) training image data arrays if split using 
        data_split method.
        -
        y_train (numpy-array) training label data arrays if split using 
        data_split method.
        -
        X_val (numpy-array) validation image data arrays if split using 
        data_split method.
        -
        y_val (numpy-array) validation label data arrays if split using 
        data_split method.
        -
        X_test (numpy-array) testing image data arrays if split using 
        data_split method.
        -
        y_test (numpy-array) testing label data arrays if split using 
        data_split method.
        -
        X_data (numpy-array): image data arrays (unsplit or merged 
        using data_merge method.).
        -
        y_data (numpy-array): data label arrays (one-hot-encoded, unsplit
        or merged using data_merge method).
        -
        min_pv (float): minimum pixel value across entire dataset.
        -
        max_pv (float): maximum pixel value across entire dataset.
        -
        size (int): total number of images in the dataset.
    
    Class Methods:
        init: constructs the necessary attributes for the dataset.
        -
        details: prints or displays summary details about the dataset.
        -
        map_classes: maps class names to label data from new list of 
        class names.
        -
        normalize: normalizes pixel values to range [0,1].
        -
        standardize: standardizes pixel values using the mean and 
        standard deviation of the training subset (note that the dataset
        must be split in order to standardize).
        -
        data_split: splits X and y data into training, validation and 
        testing subsets.
        -
        data_merge: merges X and y training and testing (and validation,
        if applicable) subsets into single dataset.
        -
        display_batch: displays random batch of images from the dataset.
        -
        save_arrays: saves dataset (or subsets) as numpy arrays in
        HDF5 format.
        -
        save_imgs: saves dataset into main directory and subdirectories 
        for each class.
        -
        augment_training_set: calls on an (initialized) imgo.augtools 
        augmenter to apply image augmentation to the Image_Dataset's 
        X_train subset.
        -
        split_rebalance: splits dataset into training and testing 
        (and validation) subsets and rebalances class sizes by calling 
        on an (initialized) imgo.augtools augmenter to generate new 
        training images (without affecting the validation/testing 
        subsets).
               
        
Module-Wide Functions
---------------------
get_class_names: fetch class names from image data directories.
-
img_to_df: compile image directories into pandas-DataFrame.
-
display_img_df: display batches of images from a pandas-DataFrame.
-
read_img_df: read images from pandas-DataFrame.
-
one_hot_encode: one-hot-encode image data labels.
-
auto_rescale: rescale image to square of specified dimensions.
-
threshold_rescale: rescale image to square of specified dimensions 
if outside threshold dimension range.
-
rescale_flow: apply either auto or threshold rescaling to images located
on local disk (with option to save).
"""

import os
import numpy as np
import pandas as pd
import random
import imageio
import cv2
import matplotlib.pyplot as plt
import h5py
from imgaug import augmenters as iaa
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from send2trash import send2trash

# ------------------------------------------------------------------------


def get_class_names(base_path):

    """
    Fetches class names from subdirectories in the directory given as the
    base path.

    Arguments:
        base_path (str): path to the directory containing images or class
        subdirectories.

    Returns:
        class_list (list): list of classes identified from subdirectories.
    """

    class_bool = 1
    for r, d, f in os.walk(base_path):
        if d == []:
            class_bool = 0
        else:
            break

    if class_bool == 0:
        class_list = []
    else:
        class_list = sorted(
            [f for f in os.listdir(base_path) if not f.startswith(".")],
            key=lambda f: f.lower(),
        )

    return class_list


# ------------------------------------------------------------------------


def img_to_df(base_path):

    """
    Fetches images and class names from subdirectories in the directory
    given as the base path and returns a DataFrame.

    Arguments:
        base_path (str): path to the directory containing images or class
        subdirectories.

    Returns:
        df (pandas-DataFrame): DataFrame of size x-by-2 (where column 0
        is the image path, column 1 is the class name, and x is the number
        of images).
    """

    class_list = get_class_names(base_path)

    if class_list == []:
        img_list = [
            f"{base_path}/{f}"
            for f in os.listdir(base_path)
            if not f.startswith(".")
        ]

        df = pd.DataFrame(img_list, columns=["image"])
        df["class"] = "no_class"
        df = df.reset_index(drop=True)
        return df
    else:
        df_list = []
        for c in class_list:
            img_list = [
                f"{base_path}/{c}/{f}"
                for f in os.listdir(f"{base_path}/{c}")
                if not f.startswith(".")
            ]
            class_df = pd.DataFrame(img_list, columns=["image"])
            class_df["class"] = c
            df_list.append(class_df)
        df = pd.concat(df_list)
        df = df.reset_index(drop=True)
        return df


# ------------------------------------------------------------------------


def display_img_df(df, batch_no, batch_size, n_rows, n_cols):

    """
    Displays images contained in an x-by-2 DataFrame (where column 0 is
    the image path, column 1 is the class name, and x is the number of
    images).

    Arguments:
        df (pandas-DataFrame): x-by-2 DataFrame (where column 0 is the
        image path, column 1 is the class name, and x is the number of
        images)
        -
        batch_size (int): size of subset (batch) of images.
        -
        batch_no (int): which batch from the DataFrame to display.
        -
        n_rows (int): number of rows of images to display.
        -
        n_cols (int): number of columns of images to display.

    Returns:
        Visualization of image batch specified.
    """

    if n_rows * n_cols != batch_size:
        raise Exception(
            f"Cannot display {batch_size} images in {n_rows} rows and {n_cols} cols."
        )

    batches = np.divmod(len(df), batch_size)[0] + bool(
        np.divmod(len(df), batch_size)[1]
    )

    if (batch_no + 1) > batches:
        raise Exception(
            f"'batch_no' argument out of range; final batch is {batches-1}."
        )

    bottom = np.arange(0, len(df), batch_size)[batch_no]
    if batch_no == batches - 1:
        top = len(df)
    else:
        top = np.arange(0, len(df), batch_size)[batch_no + 1]
    batch_df = df[bottom:top]

    img_list = []
    label_list = []
    n = 0
    for i, j in batch_df.iterrows():
        img = imageio.imread(j[0])
        img_list.append(img)
        if j[1] == "no_class":
            label = f"batch {batch_no}, img {n}"
            n += 1
        else:
            label = j[1]
        label_list.append(label)
    img_array = np.array(img_list)
    label_array = np.array(label_list)

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = "Helvetica"
    plt.rcParams["text.color"] = "#333F4B"

    fig = plt.figure(figsize=(12, 8))

    for i in range(1, (img_array.shape[0]) + 1):
        ax = fig.add_subplot(n_rows, n_cols, i)
        ax.imshow(img_array[i - 1])
        ax.set_title(label_array[i - 1])
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    plt.show()


# ------------------------------------------------------------------------


def read_img_df(df, img_scale=None, class_name=None, save=False):

    """
    Reads images contained in an x-by-2 DataFrame (where column 0 is the
    image path, column 1 is the class name, and x is the number of
    images).

    Arguments:
        df (pandas-DataFrame): x-by-2 DataFrame (where column 0 is the
        image path, column 1 is the class name, and x is the number of
        images)

    Keyword Arguments:
        img_scale (int) optional: dimensions for desired (square) output
        images. If None, no resizing will occur. Defaults to None.
        -
        class_name (str) optional: name of a class in the DataFrame. If
        given, only images belonging to that class will be read. Defaults
        to None.
        -
        save (bool) optional: whether or not to save resulting array of
        image data as a .h5 file in the path
        'imgo_output/uptools/preprocessing'. Note that images can only
        be saved if they have been rescaled using the 'img_scale'
        argument. Dafaults to False.

    Returns:
        img_array (numpy-array): images as numpy-array.
    """

    if class_name:
        data_df = df.loc[df["class"] == class_name]
    else:
        data_df = df
    img_list = []
    label_list = []
    n = 0
    for i, j in data_df.iterrows():
        raw_img = imageio.imread(j[0])
        if img_scale:
            img = auto_rescale(raw_img, img_scale)
        else:
            img = raw_img
        img_list.append(img)
        if j[1] == "no_class":
            label = n
            n += 1
        else:
            label = j[1]
        label_list.append(label)
    img_array = np.array(img_list)
    label_array = np.array(label_list)

    if save:
        if img_scale == None:
            raise Exception(
                "Cannot save images with inconsistent dimensions."
            )
        else:

            my_path = "imgo_output/uptools/preprocessing"

            r = None
            for i in my_path.split("/"):
                if r == None:
                    if not os.path.exists(i):
                        os.mkdir(i)
                    r = i
                else:
                    if not os.path.exists(r + "/" + i):
                        os.mkdir(r + "/" + i)
                    r = r + "/" + i

            with h5py.File(f"{r}/X_data.h5", "w") as hf:
                hf.create_dataset(f"X_data", data=img_array)
            print(f"{r}/X_data.h5 saved successfully.")

    return img_array


# ------------------------------------------------------------------------


def one_hot_encode(y_data, class_list, save=False):

    """
    One-hot encodes list of class labels.

    Note that the one-hot encoded data returned will be based on the
    class_list given sorted in alphabetical order.

    Arguments:
        y_data (list, tuple, or 1D-array): list of class labels.
        -
        class_list (list): list of class names against which the one-hot
        encoding occurs.

    Keyword Arguments:
        save (bool) optional: whether or not to save resulting array of
        one-hot encoded data as a .h5 file in the path
        'imgo_output/uptools/preprocessing'. Dafaults to False.

    Returns:
        y_data (numpy-array): one-hot encoded class label data as numpy-
        array.
    """

    y_list = []
    labels = []
    if type(class_list) is not list:
        raise Exception(
            f"class_list must be a list; {type(class_list)} given."
        )
    else:
        classes = sorted(list(set(class_list)), key=lambda f: f.lower())
        class_no = len(classes)

    for i in y_data:
        ohe_init = np.zeros(class_no)
        label = i
        labels.append(label)
        ohe_init[classes.index(label)] = 1
        ohe_label = ohe_init
        y_list.append(ohe_label)

    y_data = np.array(y_list)

    if save:

        my_path = "imgo_output/uptools/preprocessing"

        r = None
        for i in my_path.split("/"):
            if r == None:
                if not os.path.exists(i):
                    os.mkdir(i)
                r = i
            else:
                if not os.path.exists(r + "/" + i):
                    os.mkdir(r + "/" + i)
                r = r + "/" + i

        with h5py.File(f"{r}/y_data.h5", "w") as hf:
            hf.create_dataset(f"y_data", data=y_data)
        print(f"{r}/y_data.h5 saved successfully.")

    return y_data


# ------------------------------------------------------------------------


def auto_rescale(img, dim):

    """
    Rescales image to a square of n-by-n pixels, where n is the integer
    value given by the 'dim' argument. Rescaling is performed with cv2
    'inter cubic' interpolation if the original dimensions are smaller than
    the target dimensions, and cv2 'inter area' interpolation if the
    original dimensions are greater than the target dimensions.

    Arguments:
        img (numpy-array): original image to rescale.
        -
        dim (int): number of pixels in the target dimensions.

    Returns:
        scaled_img (numpy-array): image rescaled into square of length and
        height equal to 'dim'.
    """

    raw_dims = np.max(img.shape)

    scale = (dim, dim)

    if raw_dims < dim:
        scaled_img = cv2.resize(
            img, scale, interpolation=cv2.INTER_CUBIC
        )
    else:
        scaled_img = cv2.resize(
            img, scale, interpolation=cv2.INTER_AREA
        )

    if len(scaled_img.shape) == 2:
        scaled_img = np.expand_dims(scaled_img, 2)

    return scaled_img


# ------------------------------------------------------------------------


def threshold_rescale(img, lower=None, upper=None):

    """
    Rescales image to a square of n-by-n pixels if the square root of the
    product of the image dimensions are lower or higher than the
    respective threshold values given by the 'lower' and 'upper' arguments.
    Rescaling is performed with cv2 'inter cubic' interpolation if the
    original dimensions are smaller than the target dimensions, and cv2
    'inter area' interpolation if the original dimensions are greater than
    the target dimensions.

    Arguments:
        img (numpy-array): original image to rescale.

    Keyword Arguments:
        lower (int) optional: the lower bound of the threshold. Defaults
        to None.
        -
        uppper (int) optional: the upper bound of the threshold. Defaults
        to None.

    Returns:
        scaled_img (numpy-array): image rescaled into square of length and
        height equal to 'lower' or 'upper', depending on which is given.
    """

    img_dim = np.sqrt(img.shape[0] * img.shape[1])

    if (lower is not None) and (upper is None):
        if img_dim < lower:
            scaled_img = cv2.resize(
                img, (lower, lower), interpolation=cv2.INTER_CUBIC
            )
        else:
            scaled_img = img
    elif (lower is None) and (upper is not None):
        if img_dim > upper:
            scaled_img = cv2.resize(
                img, (upper, upper), interpolation=cv2.INTER_AREA
            )
        else:
            scaled_img = img
    elif (lower is not None) and (upper is not None):
        if img_dim < lower:
            scaled_img = cv2.resize(
                img, (lower, lower), interpolation=cv2.INTER_CUBIC
            )
        elif img_dim > upper:
            scaled_img = cv2.resize(
                img, (upper, upper), interpolation=cv2.INTER_AREA
            )
        else:
            scaled_img = img
    else:
        scaled_img = img

    if len(scaled_img.shape) == 2:
        scaled_img = np.expand_dims(scaled_img, 2)

    return scaled_img


# ------------------------------------------------------------------------


def rescale_flow(
    base_path,
    rescale_mode,
    dim=None,
    lower=None,
    upper=None,
    class_selection=None,
    save=False,
):

    """
    Fetches images and class names from subdirectories in the directory
    given as the base path and rescales the images using either the
    'auto_rescale' or the 'threshold_rescale' function, with the option
    to save the rescaled images in place of the original images.

    Arguments:
        base_path (str): path to the directory containing images or class
        subdirectories.
        -
        rescale_mode (str): which rescale function to use; either 'auto'
        or 'threshold'.

    Keyword Arguments:
        dim (int) optional: number of pixels in the target dimensions.
        Defaults to None.
        -
        lower (int) optional: the lower bound of the threshold. Defaults
        to None.
        -
        uppper (int) optional: the upper bound of the threshold. Defaults
        to None.
        -
        class_selection (list) optional: list of class names on which
        to perform the rescaling. If not given, will apply to all the
        identified in the directories. Defaults to None.
        -
        save (bool) optional: whether or not to save the rescaled images
        in the directories from which they were drawn. Note that saving
        will overwrite the original images. Defaults to None.

    Returns:
        X (numpy-array): images in array form (if 'save' is False).
        -
        y (numpy-array): one-hot encoded label data (if 'save' is False).
    """

    if rescale_mode not in ["auto", "threshold"]:
        raise Exception(
            "Choose valid rescale mode: 'auto' or 'threshold'."
        )

    df = img_to_df(base_path)
    scaled_imgs = []
    scaled_imgs_labels = []

    if class_selection:
        if type(class_selection) is not list:
            raise Exception(
                f"Class selection must be a list; {type(class_selection)} given."
            )
        else:
            df = df.loc[df["class"].isin(class_selection)]

    class_list = sorted(
        list(df["class"].unique()), key=lambda f: f.lower()
    )

    for i, j in tqdm(df.iterrows(), total=len(df)):

        img = imageio.imread(j[0])
        if rescale_mode == "auto":
            scaled_img = auto_rescale(img, dim)
        elif rescale_mode == "threshold":
            scaled_img = threshold_rescale(
                img, lower=lower, upper=upper
            )
        else:
            scaled_img = img

        if len(scaled_img.shape) == 2:
            scaled_img = np.expand_dims(scaled_img, 2)

        if save:
            imageio.imwrite(j[0], scaled_img)
        else:
            scaled_imgs.append(scaled_img)
            scaled_imgs_labels.append(j[1])

    if not save:

        X = np.array(scaled_imgs)
        y = one_hot_encode(scaled_imgs_labels, class_list)

        return X, y


# ------------------------------------------------------------------------


class Image_Dataset:

    """
    Image_Dataset: Class representing an image dataset, being a collection
    of X (square image data) and y (label data) arrays.

    Attributes
    ----------
    base_path (str): path to the directory containing images or class
    subdirectories.
    -
    mode (str): format of source image data: "imgs" if raw images, "np" if
    numpy-arrays, or "h5" if HDF5 format.
    -
    reduce (str): statistical reduction performed on raw data: "norm" for
    pixel value normalization, "std" for pixel value standardization
    (using training data statistics). Note that any reduction will be
    undone if the datasets are saved to disk.
    -
    class_list (list): list of classes in the dataset.
    -
    class_no (int): number of classes in the dataset.
    -
    split (int): number of splits in the dataset: 2 if split into
    training, validation, and testing subsets; 1 if split into
    training and testing subsets; and 0 if not split (or merged).
    -
    dims (tuple): dimensions of the images in the dataset (set to
    "various" (str) if the images are not of a consistent size).
    -
    img_shape (tuple): dimensions and channels of the images in the
    dataset.
    -
    expand (str): statistical expansion performed on raw data: "de_norm"
    for de-normalization of pixel values, and "de_std" for
    de-standardization of pixel values.
    -
    shadow (dict): image and label data for each data subset "train",
    "val", "test", and "data" (if unsplit or merged), in integer (ie
    non-normalized and non-standardized form).
    -
    X_train (numpy-array) training image data arrays if split using
    data_split method.
    -
    y_train (numpy-array) training label data arrays if split using
    data_split method.
    -
    X_val (numpy-array) validation image data arrays if split using
    data_split method.
    -
    y_val (numpy-array) validation label data arrays if split using
    data_split method.
    -
    X_test (numpy-array) testing image data arrays if split using
    data_split method.
    -
    y_test (numpy-array) testing label data arrays if split using
    data_split method.
    -
    X_data (numpy-array): image data arrays (unsplit or merged using
    data_merge method.).
    -
    y_data (numpy-array): data label arrays (one-hot-encoded, unsplit
    or merged using data_merge method).
    -
    min_pv (float): minimum pixel value across entire dataset.
    -
    max_pv (float): maximum pixel value across entire dataset.
    -
    size (int): total number of images in the dataset.

    Methods
    -------
    init: constructs the necessary attributes for the dataset.
    -
    details: prints or displays summary details about the dataset.
    -
    map_classes: maps class names to label data from new list of class
    names.
    -
    normalize: normalizes pixel values to range [0,1].
    -
    standardize: standardizes pixel values using the mean and standard
    deviation of the training subset (note that the dataset must be split
    in order to standardize).
    -
    data_split: splits X and y data into training, validation and testing
    subsets.
    -
    data_merge: merges X and y training and testing (and validation, if
    applicable) subsets into single dataset.
    -
    display_batch: displays random batch of images from the dataset.
    -
    save_arrays: saves dataset (or subsets) as numpy arrays in HDF5
    format.
    -
    save_imgs: saves dataset into main directory and subdirectories for
    each class.
    -
    augment_training_set: calls on an (initialized) imgo.augtools
    augmenter to apply image augmentation to the Image_Dataset's X_train
    subset.
    -
    split_rebalance: splits dataset into training and testing
    (and validation) subsets and rebalances class sizes by calling on
    an (initialized) imgo.augtools augmenter to generate new training
    images (without affecting the validation/testing subsets).
    """

    def __init__(
        self,
        base_path,
        mode,
        img_scale,
        pre_norm=False,
        pre_std=False,
        normalize=False,
        standardize=False,
        manual_classes=None,
    ):

        """
        Constructs all the necessary attributes for the dataset.

        Arguments:
            base_path (str): path to the directory containing images or
            class subdirectories.
            -
            mode (str): format of source image data: "imgs" if raw images,
            "np" if numpy-arrays, or "h5" if HDF5 format.
            -
            img_scale (int): dimensions for desired (square) output
            images.

        Keyword Arguments:
            pre_norm (bool) optional: whether or not the numpy data has
            been normalized prior to initialization of the Image_Dataset
            object. If it has been normalized, not setting this argument
            to True will result in error. Defaults to False.
            -
            pre_std (bool) optional: whether or not the numpy data has
            been standardized prior to initialization of the Image_Dataset
            object. If it has been normalized, not setting this argument
            to True will result in error. Defaults to False.
            -
            normalize (bool) optional: whether or not to normalize image
            pixel values to range [0,1]. Note that normalized datasets
            will be saved in non-normalized form if saved to disk.
            Defaults to False.
            -
            standardize (bool) optional: whether or not to standardize
            the pixel values in the training and testing (and validation)
            sets using the mean and standard deviation of the training
            data. Note that the standardization operation will occur only
            when the dataset is split using the 'data_split' method. Note
            also that standardized datasets will be saved in
            non-standardized form if saved to disk. Defaults to False.
            -
            manual_classes (list) optional: list of class names if using
            "np" or "h5" modes. Classes in this list will be tagged onto
            the y data in alphabetical order (ie column 0 of the y data
            will be named as the first class in the list when sorted
            alphabetically).

        Yields:
            Image_Dataset object with attributes as specified.
        """

        self.base_path = base_path

        if mode in ["imgs", "np", "h5"]:
            self.mode = mode
        else:
            raise Exception(
                "Must select valid mode: 'imgs', 'np', or 'h5'."
            )

        if type(img_scale) is int:
            rescale_dims = img_scale
        else:
            raise Exception(
                f"'img_scale' argument must be integer, {type(img_scale)} given."
            )
            rescale_dims = None

        if normalize and standardize:
            raise Exception(
                "Can either normalize or standardize data, cannot do both."
            )
            self.reduce = None
        elif normalize and (not standardize):
            self.reduce = "norm"
        elif (not normalize) and standardize:
            self.reduce = "std"
        else:
            self.reduce = None

        class_list_pending = False

        if manual_classes is not None:
            if mode == "imgs":
                raise Exception(
                    "'manual_classes' argument can only be given for 'np' and 'h5' modes."
                )
                self.class_list = None
                self.class_no = None
            else:
                if type(manual_classes) is list:
                    self.class_list = sorted(
                        [str(i) for i in manual_classes],
                        key=lambda f: f.lower(),
                    )
                    self.class_no = len(self.class_list)
                else:
                    raise Exception(
                        f"'manual_classes' argument must be list, {type(manual_classes)} given."
                    )
                    self.class_list = None
                    self.class_no = None
        else:
            if self.mode == "imgs":
                self.class_list = get_class_names(self.base_path)
                if self.class_list == []:
                    self.class_list.append("no_class")
                self.class_no = len(self.class_list)
            else:
                class_list_pending = True
                self.class_list = None
                self.class_no = None

        combo_sets = {
            "train": [None, None],
            "val": [None, None],
            "test": [None, None],
            "data": [None, None],
        }

        if (self.mode == "np") or (self.mode == "h5"):

            file_list = []
            for r, d, f in os.walk(base_path):
                for i in f:
                    if not i.startswith("."):
                        file_list.append(
                            [i, os.path.relpath(os.path.join(r, i))]
                        )

            for file in file_list:
                if file[0].lower().endswith(".npz"):
                    for k, v in combo_sets.items():
                        if file[0].lower().startswith(f"x_{k}".lower()):
                            data_load = np.load(
                                file[1], allow_pickle=True
                            )
                            combo_sets[k][0] = data_load[
                                data_load.files[0]
                            ]
                        elif (
                            file[0].lower().startswith(f"y_{k}".lower())
                        ):
                            data_load = np.load(
                                file[1], allow_pickle=True
                            )
                            combo_sets[k][1] = data_load[
                                data_load.files[0]
                            ]

                elif file[0].lower().endswith(".npy"):
                    for k, v in combo_sets.items():
                        if file[0].lower().startswith(f"x_{k}".lower()):
                            combo_sets[k][0] = np.load(
                                file[1], allow_pickle=True
                            )
                        elif (
                            file[0].lower().startswith(f"y_{k}".lower())
                        ):
                            combo_sets[k][1] = np.load(
                                file[1], allow_pickle=True
                            )

                elif file[0].lower().endswith(".h5"):
                    for k, v in combo_sets.items():
                        if file[0].lower().startswith(f"x_{k}".lower()):
                            with h5py.File(f"{file[1]}", "r") as hf:
                                combo_sets[k][0] = hf[f"X_{k}"][:]
                        elif (
                            file[0].lower().startswith(f"y_{k}".lower())
                        ):
                            with h5py.File(f"{file[1]}", "r") as hf:
                                combo_sets[k][1] = hf[f"y_{k}"][:]

                else:
                    raise Exception(
                        "No valid '.npy', '.npz', or '.h5' files identified."
                    )

            if class_list_pending:
                y_shapes = {}
                for v in [
                    v
                    for k, v in combo_sets.items()
                    if (v[1] is not None)
                ]:
                    y_shapes[k] = v[1].shape[1]
                if len(set(y_shapes.values())) == 1:
                    clist = []
                    self.class_no = list(set(y_shapes.values()))[0]
                    c = str(0) * len(str(self.class_no))
                    for i in range(1, self.class_no + 1):
                        len_dif = len(str(self.class_no)) - (
                            len(c + str(i)) - len(str(self.class_no))
                        )
                        cl = (str(0) * len_dif) + str(i)
                        clist.append(f"class_{cl}")
                    self.class_list = clist
                else:
                    raise Exception(
                        "Inconsistent number of classes in y-arrays."
                    )

        elif self.mode == "imgs":
            data_df = img_to_df(self.base_path)

            X_list = []
            y_list = []

            for i, j in tqdm(
                data_df.iterrows(),
                total=len(data_df),
                desc="Reading images",
                position=0,
            ):
                ohe_init = np.zeros(self.class_no)
                label = j[1]
                ohe_init[self.class_list.index(label)] = 1
                ohe_label = ohe_init
                y_list.append(ohe_label)

                img_data = imageio.imread(j[0])

                if rescale_dims:
                    img = auto_rescale(img_data, rescale_dims)
                else:
                    img = img_data
                X_list.append(img)

            combo_sets["data"][0] = np.array(X_list)
            combo_sets["data"][1] = np.array(y_list)

        self.split = int(
            len([1 for k, v in combo_sets.items() if v[0] is not None])
            - 1
        )

        x_eqdims = []
        for k, v in combo_sets.items():
            if v[0] is not None:
                if (
                    (len(v[0].shape) == 4) or (len(v[0].shape) == 3)
                ) and (v[0].shape[1] == v[0].shape[2]):
                    x_eqdims.append(v[0].shape[1])

        if (len(set(x_eqdims)) == 1) and (
            len(x_eqdims) - 1 == self.split
        ):
            self.dims = (list(set(x_eqdims))[0], list(set(x_eqdims))[0])
        else:
            self.dims = "various"

        if pre_norm and pre_std:
            raise Exception(
                "Cannot expand data if both pre-normalized and pre-standardized."
            )
            self.expand = None

        elif pre_norm and (not pre_std):
            self.expand = "de_norm"

        elif (not pre_norm) and pre_std:
            if self.dims == "various":
                raise Exception(
                    "Cannot expand data if image dimensions are not the same."
                )
            else:
                if self.split == 0:
                    raise Exception(
                        "Cannot de-standardize unsplit data."
                    )
                    self.expand = None
                else:
                    self.expand = "de_std"
        else:
            self.expand = None

        X_sets = {"train": [], "val": [], "test": [], "data": []}
        min_pv = None
        max_pv = None

        for k, v in X_sets.items():
            if combo_sets[k][0] is not None:
                for i in tqdm(
                    np.arange(combo_sets[k][0].shape[0]),
                    total=combo_sets[k][0].shape[0],
                    desc=f"Processing X_{k}",
                ):

                    if self.expand is not None:
                        if self.expand == "de_std":
                            img_min = np.min(combo_sets[k][0][i])
                            img_max = np.max(combo_sets[k][0][i])
                            raw_img = (
                                ((combo_sets[k][0][i] - img_min) * 255)
                                / (img_max - img_min)
                            ).astype(np.uint8)
                            clean_img = np.clip(raw_img, 0, 255).astype(
                                np.uint8
                            )
                        elif self.expand == "de_norm":
                            raw_img = (
                                combo_sets[k][0][i] * 255
                            ).astype(np.uint8)
                            clean_img = np.clip(raw_img, 0, 255).astype(
                                np.uint8
                            )
                        else:
                            raw_img = combo_sets[k][0][i]
                            clean_img = np.clip(raw_img, 0, 255).astype(
                                np.uint8
                            )
                    else:
                        raw_img = combo_sets[k][0][i]
                        clean_img = np.clip(raw_img, 0, 255).astype(
                            np.uint8
                        )

                    if self.mode == "imgs":
                        img = clean_img

                    else:
                        if rescale_dims:
                            img = auto_rescale(clean_img, rescale_dims)
                            self.dims = (rescale_dims, rescale_dims)
                        else:
                            img = clean_img

                    v.append(img)
                    img_min = np.min(img)
                    img_max = np.max(img)

                    if min_pv is None:
                        min_pv = img_min
                    else:
                        if img_min < min_pv:
                            min_pv = img_min

                    if max_pv is None:
                        max_pv = img_max
                    else:
                        if img_max > max_pv:
                            max_pv = img_max

                combo_sets[k][0] = np.array(v)

        for k, v in combo_sets.items():
            if v[0] is not None:
                if len(v[0].shape) == 3:
                    v[0] = np.expand_dims(v[0], 3)

        channels = []
        for k, v in combo_sets.items():
            if v[0] is not None:
                channels.append(v[0].shape[-1])
        if len(set(channels)) != 1:
            raise Exception(
                "Cannot create ImageDataset with inconsistent number of channels."
            )
        channels = list(set(channels))[0]

        self.img_shape = (self.dims[0], self.dims[1], channels)

        self.shadow = combo_sets

        if self.reduce == "std":
            if self.dims == "various":
                raise Exception(
                    "Cannot standardize data if image dimensions are not the same."
                )
            else:
                if self.split == 0:
                    print(
                        "Data will be standardized when split using 'data_split'."
                    )
                    for k, v in combo_sets.items():
                        if v[0] is None:
                            setattr(self, f"X_{k}", None)
                        else:
                            setattr(self, f"X_{k}", v[0])
                        if v[1] is None:
                            setattr(self, f"y_{k}", None)
                        else:
                            setattr(self, f"y_{k}", v[1])
                    self.min_pv = min_pv
                    self.max_pv = max_pv
                else:
                    print("Standardizing...")
                    self.mu = (
                        np.sum(combo_sets["train"][0])
                        / combo_sets["train"][0].size
                    )
                    self.sigma = np.sqrt(
                        np.sum((combo_sets["train"][0] - self.mu) ** 2)
                        / combo_sets["train"][0].size
                    )
                    for k, v in combo_sets.items():
                        if v[0] is None:
                            setattr(self, f"X_{k}", None)
                        else:
                            setattr(
                                self,
                                f"X_{k}",
                                (v[0] - self.mu) / self.sigma,
                            )
                        if v[1] is None:
                            setattr(self, f"y_{k}", None)
                        else:
                            setattr(self, f"y_{k}", v[1])

                    self.min_pv = (min_pv - self.mu) / self.sigma
                    self.max_pv = (max_pv - self.mu) / self.sigma

        elif self.reduce == "norm":
            print("Normalizing...")
            for k, v in combo_sets.items():
                if v[0] is None:
                    setattr(self, f"X_{k}", None)
                else:
                    setattr(self, f"X_{k}", v[0] / 255)
                if v[1] is None:
                    setattr(self, f"y_{k}", None)
                else:
                    setattr(self, f"y_{k}", v[1])

            self.min_pv = min_pv / 255
            self.max_pv = max_pv / 255

        else:
            for k, v in combo_sets.items():
                if v[0] is None:
                    setattr(self, f"X_{k}", None)
                else:
                    setattr(self, f"X_{k}", v[0])
                if v[1] is None:
                    setattr(self, f"y_{k}", None)
                else:
                    setattr(self, f"y_{k}", v[1])

            self.min_pv = min_pv
            self.max_pv = max_pv

        if self.split == 0:
            self.size = self.y_data.shape[0]
        elif self.split == 1:
            self.size = self.y_train.shape[0] + self.y_test.shape[0]
        else:
            self.size = (
                self.y_train.shape[0]
                + self.y_val.shape[0]
                + self.y_test.shape[0]
            )

        print("Image_Datset initialized successfully.")

    #     ----------

    def details(self, plot=False):

        """
        Prints summary details of Image_Dataset object, or displays the
        details as a visualization if kwarg 'plot' is given as True.
        """

        labels = {}
        labs_nums = {}

        for k, v in self.shadow.items():
            if v[1] is not None:
                labels[k] = []
                for i in np.arange(v[1].shape[0]):
                    label_index = np.argmax(v[1][i], axis=0)
                    label = self.class_list[label_index]
                    labels[k].append(label)

        for k, v in labels.items():
            lab, num = np.unique(labels[k], return_counts=True)
            labs_nums[k] = dict(zip(list(lab), list(num)))

        if self.split == 0:
            imgs_per_class = labs_nums["data"]
            df_cols = ["data"]
            splits = None
            colors = ["#81ecec"]
        elif self.split == 1:
            imgs_per_class = labs_nums
            df_cols = ["train", "test"]
            splits = df_cols
            colors = ["#81ecec", "#a29bfe"]
        else:
            imgs_per_class = labs_nums
            df_cols = ["train", "val", "test"]
            splits = df_cols
            colors = ["#81ecec", "#74b9ff", "#a29bfe"]

        val_ranges = {"min": self.min_pv, "max": self.max_pv}

        ds_dict = {
            "total_images": self.size,
            "splits": splits,
            "images_per_class": imgs_per_class,
            "image_shape": self.img_shape,
            "pixel_values": val_ranges,
        }

        if plot:

            ldf = pd.DataFrame(labs_nums, columns=df_cols).fillna(0)
            ldf = ldf.iloc[::-1]

            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = "Helvetica"
            plt.rcParams["axes.edgecolor"] = "#333F4B"
            plt.rcParams["axes.linewidth"] = 0.8
            plt.rcParams["xtick.color"] = "#333F4B"
            plt.rcParams["ytick.color"] = "#333F4B"
            plt.rcParams["text.color"] = "#333F4B"

            fig, ax = plt.subplots(figsize=(10, 6))
            fig.text(
                0,
                0.9,
                "Class",
                fontsize=15,
                fontweight="black",
                color="#333F4B",
            )

            ldf.plot.barh(stacked=True, ax=ax, color=colors)

            ax.set_xlabel(
                "Images",
                fontsize=15,
                fontweight="black",
                color="#333F4B",
            )
            ax.set_ylabel("")
            ax.tick_params(axis="both", which="major", labelsize=12)
            if self.split == 0:
                ax.legend().set_visible(False)
            else:
                ax.legend(bbox_to_anchor=(1, 0.5))
            ax.spines["top"].set_color("none")
            ax.spines["right"].set_color("none")
            ax.spines["left"].set_smart_bounds(True)
            ax.spines["bottom"].set_smart_bounds(True)

            plt.show()

        else:

            print("Image_Dataset details")
            print("---------------------")
            for k, v in {
                i: ds_dict[i]
                for i in ds_dict
                if i != "images_per_class"
            }.items():
                print(f"{k:<20}{v}\n---")
            print("images_per_class\n-")
            for k, v in ds_dict["images_per_class"].items():
                print(f"{k:<20}{v}\n-")
            if self.split != 0:
                totals = {}
                for c in self.class_list:
                    c_total = []
                    for k, v in ds_dict["images_per_class"].items():
                        c_total.append(
                            ds_dict["images_per_class"][k][c]
                        )
                        totals[c] = sum(c_total)
                print(f"{'totals':<20}{totals}")

    #     ----------

    def map_classes(self, class_list):

        """
        Maps class names from a given list of class names.
        """

        if type(class_list) is list:
            new_list = sorted(
                [str(i) for i in class_list], key=lambda f: f.lower()
            )

            if len(new_list) == len(self.class_list):
                for i in range(len(self.class_list)):
                    self.class_list[i] = new_list[i]
            else:
                raise Exception(
                    "Number of classes given does not match number in Image_Dataset."
                )
        else:
            raise Exception(
                f"'class_list' argument must be list, {type(class_list)} given."
            )

    #     ----------

    def normalize(self):

        """
        Normalizes pixel values to range [0,1].
        """

        if self.reduce == "norm":
            raise Exception("Data has already been normalized.")
        elif self.reduce == "std":
            raise Exception("Cannot normalize standardized data.")
        else:
            subsets = self.shadow
            print("Normalizing...")
            for k, v in subsets.items():
                if v[0] is None:
                    setattr(self, f"X_{k}", None)
                else:
                    setattr(self, f"X_{k}", v[0] / 255)
                if v[1] is None:
                    setattr(self, f"y_{k}", None)
                else:
                    setattr(self, f"y_{k}", v[1])

            self.min_pv = self.min_pv / 255
            self.max_pv = self.max_pv / 255
            self.reduce = "norm"
            print("Normalization complete.")

    #     ----------

    def standardize(self):

        """
        Standardizes pixel values using the mean and standard deviation of
        the training subset (note that the dataset must be split in order
        to standardize).
        """

        if self.reduce == "std":
            raise Exception("Data has already been standardized.")
        elif self.reduce == "norm":
            raise Exception("Cannot standardize normalized data.")
        else:
            subsets = self.shadow

            if self.dims == "various":
                raise Exception(
                    "Cannot standardize data if image dimensions are not the same."
                )
            elif self.split == 0:
                raise Exception("Cannot standardize unsplit data.")
            else:
                print("Standardizing...")
                self.mu = (
                    np.sum(subsets["train"][0])
                    / subsets["train"][0].size
                )
                self.sigma = np.sqrt(
                    np.sum((subsets["train"][0] - self.mu) ** 2)
                    / subsets["train"][0].size
                )
                for k, v in subsets.items():
                    if v[0] is None:
                        setattr(self, f"X_{k}", None)
                    else:
                        setattr(
                            self,
                            f"X_{k}",
                            (v[0] - self.mu) / self.sigma,
                        )
                    if v[1] is None:
                        setattr(self, f"y_{k}", None)
                    else:
                        setattr(self, f"y_{k}", v[1])

                self.min_pv = (self.min_pv - self.mu) / self.sigma
                self.max_pv = (self.max_pv - self.mu) / self.sigma
                self.reduce = "std"
                print("Standardization complete.")

    #     ----------

    def data_split(self, split_ratio, seed=None, stratify=False):

        """
        Splits the Image_Dataset object into training and testing, and/or
        validation subsets.

        Note that this method is built using Scikit-Learn's
        train_test_split. For more information see:
        https://scikit-learn.org/stable/modules/classes.html#

        Arguments:
            split_ratio (tuple): ratios in the form (a, b, c) used
            to split the dataset; where a, b, and c are float values
            representing the desired proportions of training,
            validation, and testing subsets, repectively; and
            a + b + c = 1. If only two values are given, ie in the
            form (a, b); the dataset will be split into training and
            testing subsets only. In this case, a + b must be equal
            to 1.

        Keyword Arguments:
            seed (int) optional: random seed for use in the data split.
            Defaults to None.
            -
            stratify (bool) optional: whether or not to preserve the class
            balances that exist in the un-split data. Defaults to False.

        Yields:
            Training, testing (and validation) subset arrays as 'X_train'
            and 'y_train', 'X_test' and 'y_test' (and 'X_val' and 'y_val'
            if 3 values passed into the 'split_ratio' argument) attributes
            of the Image_Dataset object. None as 'X_data' and 'y_data'
            attributes of the Image_Dataset object.
        """

        if self.split != 0:
            raise Exception(
                "Cannot split dataset that has already been split."
            )
        else:

            if (type(split_ratio) is list) or (
                type(split_ratio) is tuple
            ):
                split_ratio = [np.round(i, 2) for i in split_ratio]
                if sum(split_ratio) == 1:
                    print("Splitting...")
                    if len(split_ratio) == 3:
                        if stratify:
                            (
                                self.shadow["train"][0],
                                Xtv,
                                self.shadow["train"][1],
                                ytv,
                            ) = train_test_split(
                                self.shadow["data"][0],
                                self.shadow["data"][1],
                                test_size=round(
                                    split_ratio[1] + split_ratio[2], 2
                                ),
                                stratify=self.shadow["data"][1],
                                random_state=seed,
                            )
                            (
                                self.shadow["val"][0],
                                self.shadow["test"][0],
                                self.shadow["val"][1],
                                self.shadow["test"][1],
                            ) = train_test_split(
                                Xtv,
                                ytv,
                                test_size=split_ratio[2]
                                / round(
                                    split_ratio[1] + split_ratio[2], 2
                                ),
                                stratify=ytv,
                                random_state=seed,
                            )
                        else:
                            (
                                self.shadow["train"][0],
                                Xtv,
                                self.shadow["train"][1],
                                ytv,
                            ) = train_test_split(
                                self.shadow["data"][0],
                                self.shadow["data"][1],
                                test_size=round(
                                    split_ratio[1] + split_ratio[2], 2
                                ),
                                random_state=seed,
                            )
                            (
                                self.shadow["val"][0],
                                self.shadow["test"][0],
                                self.shadow["val"][1],
                                self.shadow["test"][1],
                            ) = train_test_split(
                                Xtv,
                                ytv,
                                test_size=split_ratio[2]
                                / round(
                                    split_ratio[1] + split_ratio[2], 2
                                ),
                                random_state=seed,
                            )
                        del Xtv
                        del ytv
                        self.shadow["data"][0] = None
                        self.shadow["data"][1] = None
                        self.split = 2

                    elif len(split_ratio) == 2:
                        if stratify:
                            (
                                self.shadow["train"][0],
                                self.shadow["test"][0],
                                self.shadow["train"][1],
                                self.shadow["test"][1],
                            ) = train_test_split(
                                self.shadow["data"][0],
                                self.shadow["data"][1],
                                test_size=split_ratio[1],
                                stratify=self.shadow["data"][1],
                                random_state=seed,
                            )
                        else:
                            (
                                self.shadow["train"][0],
                                self.shadow["test"][0],
                                self.shadow["train"][1],
                                self.shadow["test"][1],
                            ) = train_test_split(
                                self.shadow["data"][0],
                                self.shadow["data"][1],
                                test_size=split_ratio[1],
                                random_state=seed,
                            )
                        self.shadow["data"][0] = None
                        self.shadow["data"][1] = None
                        self.shadow["val"][0] = None
                        self.shadow["val"][1] = None
                        self.split = 1
                    else:
                        raise Exception(
                            "'split_ratio' argument must be list or tuple of 2 or 3 floats."
                        )
                else:
                    raise Exception(
                        "'split_ratio' argument must sum to 1."
                    )
            else:
                raise Exception(
                    f"'split_ratio' argument must be list or tuple, {type(split_ratio)} given."
                )

            subsets = self.shadow

            if self.reduce == "std":
                if self.dims == "various":
                    raise Exception(
                        "Cannot standardize data if image dimensions are not the same."
                    )
                else:
                    print("Standardizing...")
                    self.mu = (
                        np.sum(subsets["train"][0])
                        / subsets["train"][0].size
                    )
                    self.sigma = np.sqrt(
                        np.sum((subsets["train"][0] - self.mu) ** 2)
                        / subsets["train"][0].size
                    )
                    for k, v in subsets.items():
                        if v[0] is None:
                            setattr(self, f"X_{k}", None)
                        else:
                            setattr(
                                self,
                                f"X_{k}",
                                (v[0] - self.mu) / self.sigma,
                            )
                        if v[1] is None:
                            setattr(self, f"y_{k}", None)
                        else:
                            setattr(self, f"y_{k}", v[1])

                    self.min_pv = (self.min_pv - self.mu) / self.sigma
                    self.max_pv = (self.max_pv - self.mu) / self.sigma

            elif self.reduce == "norm":
                print("Normalizing...")
                for k, v in subsets.items():
                    if v[0] is None:
                        setattr(self, f"X_{k}", None)
                    else:
                        setattr(self, f"X_{k}", v[0] / 255)
                    if v[1] is None:
                        setattr(self, f"y_{k}", None)
                    else:
                        setattr(self, f"y_{k}", v[1])

            else:
                for k, v in subsets.items():
                    if v[0] is None:
                        setattr(self, f"X_{k}", None)
                    else:
                        setattr(self, f"X_{k}", v[0])
                    if v[1] is None:
                        setattr(self, f"y_{k}", None)
                    else:
                        setattr(self, f"y_{k}", v[1])

            if self.split == 1:
                print(
                    "Data sucessfully split into training and testing subsets."
                )
            elif self.split == 2:
                print(
                    "Data sucessfully split into training, validation, and testing subsets."
                )

    #     ----------

    def data_merge(self):

        """
        Merges a split Image_Dataset object into a single dataset.
        Yields full X and y data arrays as 'X_data' and 'y_data'
        attributes of the Image_Dataset object, and None as the 'X_train',
        'y_train', 'X_val', 'y_val', 'X_test', and 'y_test' attributes of
        the Image_Dataset object.
        """

        if self.split == 0:
            raise Exception(
                "Cannot merge dataset that has not been split."
            )
        else:
            print("Merging...")
            x_merge = [
                v[0]
                for k, v in self.shadow.items()
                if (v[0] is not None)
            ]
            y_merge = [
                v[1]
                for k, v in self.shadow.items()
                if (v[1] is not None)
            ]
            try:
                self.shadow["data"][0] = np.concatenate(
                    [
                        v[0]
                        for k, v in self.shadow.items()
                        if (v[0] is not None)
                    ]
                )
                self.shadow["data"][1] = np.concatenate(
                    [
                        v[1]
                        for k, v in self.shadow.items()
                        if (v[1] is not None)
                    ]
                )
            except:
                merge_x_listwise = []
                merge_y_listwise = []
                for i in x_merge:
                    for x in np.arange(i.shape[0]):
                        merge_x_listwise.append(i[x])
                for i in y_merge:
                    for y in np.arange(i.shape[0]):
                        merge_y_listwise.append(i[y])

                self.shadow["data"][0] = np.array(merge_x_listwise)
                self.shadow["data"][1] = np.array(merge_y_listwise)

            self.shadow["train"] = [None, None]
            self.shadow["val"] = [None, None]
            self.shadow["test"] = [None, None]
            self.split = 0

            subsets = self.shadow

            if self.reduce == "std":
                if self.dims == "various":
                    raise Exception(
                        "Cannot standardize data if image dimensions are not the same."
                    )
                else:
                    print(
                        "Note: merged dataset will not be standardized."
                    )
                    for k, v in subsets.items():
                        if v[0] is None:
                            setattr(self, f"X_{k}", None)
                        else:
                            setattr(self, f"X_{k}", v[0])
                        if v[1] is None:
                            setattr(self, f"y_{k}", None)
                        else:
                            setattr(self, f"y_{k}", v[1])

                    self.min_pv = (
                        (self.min_pv * self.sigma) + self.mu
                    ).astype(np.uint8)
                    self.max_pv = (
                        (self.max_pv * self.sigma) + self.mu
                    ).astype(np.uint8)

            elif self.reduce == "norm":
                print("Normalizing...")
                for k, v in subsets.items():
                    if v[0] is None:
                        setattr(self, f"X_{k}", None)
                    else:
                        setattr(self, f"X_{k}", v[0] / 255)
                    if v[1] is None:
                        setattr(self, f"y_{k}", None)
                    else:
                        setattr(self, f"y_{k}", v[1])

            else:
                for k, v in subsets.items():
                    if v[0] is None:
                        setattr(self, f"X_{k}", None)
                    else:
                        setattr(self, f"X_{k}", v[0])
                    if v[1] is None:
                        setattr(self, f"y_{k}", None)
                    else:
                        setattr(self, f"y_{k}", v[1])

            print("Data sucessfully merged into single data set.")

    #     ----------

    def display_batch(self, n_rows, n_cols):

        """
        Displays random batch of images from the Image_Dataset object,
        along with class label and data subset if drawn from a split
        dataset.

        Arguments:
            n_rows (int): number of rows of images to display.
            -
            n_cols (int): number of columns of images to display.

        Returns:
            Visualization of random batch of images from the dataset.
        """

        if n_rows * n_cols > self.size:
            raise Exception(
                f"Cannot display {n_rows*n_cols} images because only {self.size} in dataset."
            )
        else:
            ds_array = np.arange(self.size)
            np.random.shuffle(ds_array)
            index_list = ds_array[0 : n_rows * n_cols]

            display_list = []

            for k, v in self.shadow.items():
                if v[1] is not None:
                    for i in np.arange(v[1].shape[0]):
                        if v[0][i].shape[2] == 1:
                            img = np.squeeze(v[0][i], 2)
                        else:
                            img = v[0][i]
                        label_index = np.argmax(v[1][i], axis=0)
                        label = self.class_list[label_index]
                        if self.split == 0:
                            display_list.append([img, label])
                        else:
                            display_list.append([img, f"{label} ({k})"])

            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = "Helvetica"
            plt.rcParams["text.color"] = "#333F4B"

            fig = plt.figure(figsize=(12, 8))

            for i in range(1, (n_rows * n_cols) + 1):
                ax = fig.add_subplot(n_rows, n_cols, i)
                ax.imshow(display_list[index_list[i - 1]][0])
                ax.set_title(display_list[index_list[i - 1]][1])
                ax.set_xticks([])
                ax.set_yticks([])

            fig.tight_layout()
            plt.show()

    #     ----------

    def save_arrays(self, save_dir):

        """
        Saves the dataset in HDF5 format into a directory specified by
        the 'save_dir' argument. Note that the directory will be
        created if it does not already exist, and that existing data
        within the specified directory will be overwritten.
        """

        my_path = "imgo_output/uptools/save_arrays/" + save_dir

        r = None
        for i in my_path.split("/"):
            if r == None:
                if not os.path.exists(i):
                    os.mkdir(i)
                r = i
            else:
                if not os.path.exists(r + "/" + i):
                    os.mkdir(r + "/" + i)
                r = r + "/" + i

        if self.reduce == "std":
            print("Saving non-standardized arrays.")
        elif self.reduce == "norm":
            print("Saving non-normalized arrays.")

        for k, v in self.shadow.items():
            if v[0] is not None:
                with h5py.File(f"{r}/X_{k}.h5", "w") as hf:
                    hf.create_dataset(f"X_{k}", data=v[0])
                print(f"{r}/X_{k}.h5 saved successfully.")
            if v[1] is not None:
                with h5py.File(f"{r}/y_{k}.h5", "w") as hf:
                    hf.create_dataset(f"y_{k}", data=v[1])
                print(f"{r}/y_{k}.h5 saved successfully.")

    #     ----------

    def save_imgs(self, save_dir):

        """
        Saves the dataset in image format into a directory specified
        by the 'save_dir' argument (images are saved into
        subdirectories for each class within the this directory).
        Note that the directory will be created if it does not already
        exist, and that existing data within the specified directory
        will be overwritten.
        """

        my_path = "imgo_output/uptools/save_imgs/" + save_dir

        r = None
        for i in my_path.split("/"):
            if r == None:
                if not os.path.exists(i):
                    os.mkdir(i)
                r = i
            else:
                if not os.path.exists(r + "/" + i):
                    os.mkdir(r + "/" + i)
                r = r + "/" + i

        class_counter = {}

        for c in self.class_list:
            class_counter[c] = 0
            if not os.path.exists(r + "/" + c):
                os.mkdir(r + "/" + c)

        for k, v in self.shadow.items():
            if (v[0] is not None) and (v[1] is not None):
                for i in tqdm(
                    np.arange(v[1].shape[0]),
                    total=v[1].shape[0],
                    desc="Saving",
                ):
                    img = v[0][i]
                    label_index = np.argmax(v[1][i], axis=0)
                    label = self.class_list[label_index]
                    class_counter[label] += 1
                    path = my_path + "/" + label
                    imageio.imwrite(
                        f"{path}/{label}_{class_counter[label]}.jpg",
                        img,
                    )

    #     ----------

    def augment_training_set(
        self,
        portion,
        augmenter=None,
        augment_scale=None,
        augment_type="random",
        order=None,
    ):

        """
        Calls on an (initialized) imgo.augtools augmenter to apply image
        augmentation to the Image_Dataset's X_train subset.

        Arguments:
            portion (float): float within the range [0,1]. This is the
            portion of images in the set that will be augmented.

        Keyword Arguments:
            augmenter (imgo.uptools Augmenter object) optional: the
            augmenter to apply to the images. Defaults to None.
            -
            augment_scale (int) optional: square dimensions to which the
            images are temporarily rescaled prior to augmentation (note
            that larger values result in better quality augmentations).
            The images will be rescaled back to their previous (square)
            dimensions after augmentation. Defaults to None.
            -
            augment_type (str) optional: either "random" or "simple".
            If "random", the class' "random_augment" method will be used
            for the augmentation. If "simple", the "simple_augment" method
            will be used. If None, "random_augment" is used. Defaults to
            None.
            -
            order (list) optional: list of indices (integer type) to
            determine the order in which the transformation functions are
            applied. Note that the transformation functions are ordered
            alphabetically by default. Only relevant if using "simple"
            as the "augment_type" (see above). Defaults to None.

        Returns:
            X_train_aug (numpy-array): the Image_Dataset's X_train object
            with augmented images added.

        Yields:
            Augmented images (in numpy-array form) added to the 'X_train'
            attribute of the Image_Dataset object.
        """

        from imgo import augtools

        if self.split == 0:
            raise Exception("Data has not been split.")

        else:

            if (portion >= 0) and (portion <= 1):
                n = int(np.round(self.X_train.shape[0] * (portion)))
                img_indices = np.random.choice(
                    self.X_train.shape[0], n, replace=False
                )
            else:
                raise Exception(
                    "Portion argument must be in range [0,1]."
                )

            X_aug_list = []
            y_aug_list = []

            for x in tqdm(
                np.arange(self.shadow["train"][0].shape[0]), position=0
            ):
                if x in img_indices:

                    img = self.shadow["train"][0][x]
                    label = self.shadow["train"][1][x]

                    if augment_scale:
                        scaled_img = auto_rescale(img, augment_scale)
                        if augment_type == "simple":
                            aug_scaled_img = augmenter.simple_augment(
                                scaled_img, order=order
                            )
                        else:
                            aug_scaled_img = augmenter.random_augment(
                                scaled_img
                            )
                        aug_img = auto_rescale(
                            aug_scaled_img, self.dims[0]
                        )
                    else:
                        if augment_type == "simple":
                            aug_img = augmenter.simple_augment(
                                img, order=order
                            )
                        else:
                            aug_img = augmenter.random_augment(
                                self.shadow["train"][0][x]
                            )

                    X_aug_list.append(aug_img)
                    y_aug_list.append(label)

            X_train_aug = np.array(X_aug_list)
            y_train_aug = np.array(y_aug_list)

            full_X = np.concatenate(
                (self.shadow["train"][0], X_train_aug), axis=0
            )
            full_y = np.concatenate(
                (self.shadow["train"][1], y_train_aug), axis=0
            )

            shuffle_indices = np.random.choice(
                full_X.shape[0], full_X.shape[0], replace=False
            )

            self.shadow["train"][0] = full_X[shuffle_indices]
            self.shadow["train"][1] = full_y[shuffle_indices]
            self.y_train = self.shadow["train"][1]

            if self.reduce == "std":
                if self.dims == "various":
                    raise Exception(
                        "Cannot standardize data if image dimensions are not the same."
                    )
                else:
                    print("Standardizing...")
                    self.mu = (
                        np.sum(self.shadow["train"][0])
                        / self.shadow["train"][0].size
                    )
                    self.sigma = np.sqrt(
                        np.sum((self.shadow["train"][0] - self.mu) ** 2)
                        / self.shadow["train"][0].size
                    )

                    for k, v in self.shadow.items():
                        if v[0] is None:
                            setattr(self, f"X_{k}", None)
                        else:
                            setattr(
                                self,
                                f"X_{k}",
                                (v[0] - self.mu) / self.sigma,
                            )
                        if v[1] is None:
                            setattr(self, f"y_{k}", None)
                        else:
                            setattr(self, f"y_{k}", v[1])

                    self.min_pv = (self.min_pv - self.mu) / self.sigma
                    self.max_pv = (self.max_pv - self.mu) / self.sigma

            if self.reduce == "norm":
                print("Normalizing...")
                self.X_train = self.shadow["train"][0] / 255

            else:
                self.X_train = self.shadow["train"][0]

            self.y_train = self.shadow["train"][1]

            if self.split == 1:
                print("Training data sucessfully augmented.")
                self.size = self.y_train.shape[0] + self.y_test.shape[0]
            elif self.split == 2:
                print("Training data sucessfully augmented.")
                self.size = (
                    self.y_train.shape[0]
                    + self.y_val.shape[0]
                    + self.y_test.shape[0]
                )

    #     ----------

    def split_rebalance(
        self,
        split_ratio,
        augmenter=None,
        augment_scale=None,
        augment_type=None,
        order=None,
        force=False,
    ):

        """
        Splits dataset into training and testing (and validation) subsets
        and rebalances class sizes by calling on an (initialized)
        imgo.augtools augmenter to generate new training images (without
        affecting the validation/testing subsets). The number of images
        generated for each class will depend on the ratios given by the
        'split_ratio' argument as well as the number of images already
        included in each class. The number of new images will be maximum
        possible such that the ratios are preserved and that at the total
        of the training and validation subsets are no larger than half of
        the smallest class.

        Arguments:
            split_ratio (tuple): ratios in the form (a, b, c) used
            to split the dataset; where a, b, and c are float values
            representing the desired proportions of training,
            validation, and testing subsets, repectively; and
            a + b + c = 1. If only two values are given, ie in the
            form (a, b); the dataset will be split into training and
            testing subsets only. In this case, a + b must be equal
            to 1. The number of images generated by the augmenter will
            depend on the ratios given as well as the number of images
            in each class.

        Keyword Arguments:
            augmenter (imgo.uptools Augmenter object) optional: the
            augmenter to apply to the images. Defaults to None.
            -
            augment_scale (int) optional: square dimensions to which the
            images are temporarily rescaled prior to augmentation (note
            that larger values result in better quality augmentations).
            The images will be rescaled back to their previous (square)
            dimensions after augmentation. Defaults to None.
            -
            augment_type (str) optional: either "random" or "simple".
            If "random", the class' "random_augment" method will be used
            for the augmentation. If "simple", the "simple_augment" method
            will be used. If None, "random_augment" is used. Defaults to
            None.
            -
            order (list) optional: list of indices (integer type) to
            determine the order in which the transformation functions are
            applied. Note that the transformation functions are ordered
            alphabetically by default. Only relevant if using "simple"
            as the "augment_type" (see above). Defaults to None.
            -
            force (bool) optional: whether or not to force the method to
            apply augmentation to datasets that are already balanced. The
            method will check if the classes are already balanced and
            raise an exception if not set to 'True'. Defaults to False.

        Returns:
            X_train_aug (numpy-array): the Image_Dataset's X_train object
            with augmented images (if inplace argument set to False).

        Yields:
            Augmented images (in numpy-array form) as the 'X_train'
            attribute of the Image_Dataset object (if inplace argument
            set to True).
        """

        from imgo import augtools

        if (type(split_ratio) is list) or (type(split_ratio) is tuple):
            split_ratio = [np.round(i, 2) for i in split_ratio]
            if np.round(sum(split_ratio), 2) == 1:
                if len(split_ratio) == 3:
                    tr_r = split_ratio[0]
                    va_r = split_ratio[1]
                    te_r = split_ratio[2]
                    vt_r = split_ratio[1] + split_ratio[2]
                elif len(split_ratio) == 2:
                    tr_r = split_ratio[0]
                    va_r = 0
                    te_r = split_ratio[1]
                    vt_r = split_ratio[1]
                else:
                    raise Exception(
                        "'split_ratio' argument must be list or tuple of 2 or 3 floats."
                    )
            else:
                raise Exception("'split_ratio' argument must sum to 1.")
        else:
            raise Exception(
                f"'split_ratio' argument must be list or tuple, {type(split_ratio)} given."
            )

        if self.split != 0:
            raise Exception(
                "Cannot split dataset that has already been split."
            )

        else:

            indices = {}
            for i in self.class_list:
                indices[i] = {
                    "imgs": [],
                    "to_bal": [],
                    "to_rem": [],
                    "valtest": [],
                    "val": [],
                    "test": [],
                }

            for i in np.arange(self.shadow["data"][1].shape[0]):
                for c in indices.keys():
                    if (
                        self.class_list[
                            np.argmax(self.shadow["data"][1][i], axis=0)
                        ]
                        == c
                    ):
                        indices[c]["imgs"].append(i)
            lens = []
            for c in indices.keys():
                lens.append(len(indices[c]["imgs"]))
            if all(i == lens[0] for i in lens):
                if not force:
                    raise Exception(
                        "Classes already appear to be balanced, set 'force' to 'True' to rebalance."
                    )
                else:
                    pass

            min_class_size = np.min(
                [len(indices[c]["imgs"]) for c in indices.keys()]
            )
            max_vt_size = int((min_class_size / 2) // 1)
            max_tr_size = int(((max_vt_size / vt_r) * tr_r) // 1)
            max_total_size = max_vt_size + max_tr_size

            train_indices = []
            val_indices = []
            test_indices = []

            for c in indices.keys():
                indices[c]["valtest"] = list(
                    np.random.choice(
                        indices[c]["imgs"],
                        size=max_vt_size,
                        replace=False,
                    )
                )
                indices[c]["val"] = list(
                    np.random.choice(
                        indices[c]["valtest"],
                        size=int((max_total_size * (va_r)) // 1),
                        replace=False,
                    )
                )
                indices[c]["test"] = [
                    i
                    for i in indices[c]["valtest"]
                    if i not in indices[c]["val"]
                ]
                for i in indices[c]["valtest"]:
                    indices[c]["imgs"].remove(i)
                val_indices += indices[c]["val"]
                test_indices += indices[c]["test"]

            bal_indices = []

            for c in indices.keys():
                class_tr_size = len(indices[c]["imgs"])

                inds = []

                dif = np.abs(class_tr_size - max_tr_size)
                passes = np.divmod(dif, class_tr_size)

                for i in range(passes[0]):
                    inds += list(
                        np.random.choice(
                            indices[c]["imgs"],
                            size=class_tr_size,
                            replace=False,
                        )
                    )
                inds += list(
                    np.random.choice(
                        indices[c]["imgs"],
                        size=passes[1],
                        replace=False,
                    )
                )

                if class_tr_size > max_tr_size:
                    indices[c]["to_rem"] = inds
                    for i in indices[c]["to_rem"]:
                        indices[c]["imgs"].remove(i)

                else:
                    indices[c]["to_bal"] = inds
                    bal_indices += indices[c]["to_bal"]

                train_indices += indices[c]["imgs"]

            X_train = []
            y_train = []
            X_val = []
            y_val = []
            X_test = []
            y_test = []

            data_range = np.arange(self.shadow["data"][1].shape[0])
            shuffle = np.random.choice(
                data_range, size=data_range.shape[0], replace=False
            )

            for i in shuffle:
                img = self.shadow["data"][0][i]
                label = self.shadow["data"][1][i]

                if i in train_indices:
                    X_train.append(img)
                    y_train.append(label)
                if i in val_indices:
                    X_val.append(img)
                    y_val.append(label)
                elif i in test_indices:
                    X_test.append(img)
                    y_test.append(label)

            for i in tqdm(
                bal_indices,
                total=len(bal_indices),
                desc="Rebalancing",
                position=0,
            ):

                img = self.shadow["data"][0][i]
                label = self.shadow["data"][1][i]

                if augment_scale:
                    scaled_img = auto_rescale(img, augment_scale)

                    if augment_type == "simple":
                        aug_scaled_img = augmenter.simple_augment(
                            scaled_img, order=order
                        )
                    else:
                        aug_scaled_img = augmenter.random_augment(
                            scaled_img
                        )

                    aug_img = auto_rescale(aug_scaled_img, self.dims[0])

                else:
                    if augment_type == "simple":
                        aug_img = augmenter.simple_augment(
                            img, order=order
                        )
                    else:
                        aug_img = augmenter.random_augment(img)

                X_train.append(aug_img)
                y_train.append(label)

                X_train_array = np.array(X_train)
                y_train_array = np.array(y_train)

                data_range_2 = np.arange(y_train_array.shape[0])
                shuffle_2 = np.random.choice(
                    data_range_2,
                    size=data_range_2.shape[0],
                    replace=False,
                )

            self.shadow["train"][0] = X_train_array[shuffle_2]
            self.shadow["train"][1] = y_train_array[shuffle_2]
            if len(split_ratio) == 3:
                self.shadow["val"][0] = np.array(X_val)
                self.shadow["val"][1] = np.array(y_val)
                self.shadow["test"][0] = np.array(X_test)
                self.shadow["test"][1] = np.array(y_test)
                self.split = 2
            if len(split_ratio) == 2:
                self.shadow["val"][0] = None
                self.shadow["val"][1] = None
                self.shadow["test"][0] = np.array(X_val + X_test)
                self.shadow["test"][1] = np.array(y_val + y_test)
                self.split = 1
            self.shadow["data"][0] = None
            self.shadow["data"][1] = None

            subsets = self.shadow

            if self.reduce == "std":
                if self.dims == "various":
                    raise Exception(
                        "Cannot standardize data if image dimensions are not the same."
                    )
                else:
                    print("Standardizing...")
                    self.mu = (
                        np.sum(subsets["train"][0])
                        / subsets["train"][0].size
                    )
                    self.sigma = np.sqrt(
                        np.sum((subsets["train"][0] - self.mu) ** 2)
                        / subsets["train"][0].size
                    )
                    for k, v in subsets.items():
                        if v[0] is None:
                            setattr(self, f"X_{k}", None)
                        else:
                            setattr(
                                self,
                                f"X_{k}",
                                (v[0] - self.mu) / self.sigma,
                            )
                        if v[1] is None:
                            setattr(self, f"y_{k}", None)
                        else:
                            setattr(self, f"y_{k}", v[1])

                    self.min_pv = (self.min_pv - self.mu) / self.sigma
                    self.max_pv = (self.max_pv - self.mu) / self.sigma

            elif self.reduce == "norm":
                print("Normalizing...")
                for k, v in subsets.items():
                    if v[0] is None:
                        setattr(self, f"X_{k}", None)
                    else:
                        setattr(self, f"X_{k}", v[0] / 255)
                    if v[1] is None:
                        setattr(self, f"y_{k}", None)
                    else:
                        setattr(self, f"y_{k}", v[1])

            else:
                for k, v in subsets.items():
                    if v[0] is None:
                        setattr(self, f"X_{k}", None)
                    else:
                        setattr(self, f"X_{k}", v[0])
                    if v[1] is None:
                        setattr(self, f"y_{k}", None)
                    else:
                        setattr(self, f"y_{k}", v[1])

            if self.split == 1:
                print(
                    "Data sucessfully rebalanced and split into training and testing subsets."
                )
                self.size = self.y_train.shape[0] + self.y_test.shape[0]
            elif self.split == 2:
                print(
                    "Data sucessfully rebalanced and split into training, validation, and testing subsets."
                )
                self.size = (
                    self.y_train.shape[0]
                    + self.y_val.shape[0]
                    + self.y_test.shape[0]
                )
