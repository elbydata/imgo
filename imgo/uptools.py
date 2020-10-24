"""
IMGO - Compile, process, and augment image data.
-------------------------------------------------
UPTOOLS module: 

Last updated: version 4.1.1

Classes
-------
Image_Dataset: Class representing an image dataset.
    
    Class Methods:
        init: constructs all the necessary attributes for the dataset.
        -
        generate: generates X and y data arrays.
        -
        labels: generates y data as list.
        -
        details: prints or displays summary details about the dataset.
        -
        display_batch: displays random batch of images from the dataset.
        -
        tvt_split: splits X and y data into training, validation and 
        testing subsets.
        -
        save_as_np: saves dataset (or subsets) as numpy_arrays.
        -
        save_as_imgdirs: saves dataset into main directory and 
        subdirectories for each class.
        -
        augment_training_set: calls on an (initialized) 
        imgo.augtools augmenter to apply image augmentation 
        to the Image_Dataset's X_train subset.
               
        
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
"""

import os
import numpy as np
import pandas as pd
import random
import imageio
import cv2
import matplotlib.pyplot as plt
import imageio
from imgaug import augmenters as iaa
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from send2trash import send2trash
from datetime import datetime

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
        if d == [] or d == ["augmented_images"]:
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
        try:
            if f"{base_path}/augmented_images" in img_list:
                img_list.remove(f"{base_path}/augmented_images")
        except:
            pass
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
            f"Cannot display {batch_size} images"
            + f" in {n_rows} rows and {n_cols} cols."
        )

    batches = np.divmod(len(df), batch_size)[0] + bool(
        np.divmod(len(df), batch_size)[1]
    )

    if (batch_no + 1) > batches:
        raise Exception(
            f"batch_no out of range;" + f" final batch is {batches-1}."
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


def read_img_df(df, class_name=None, save=False):

    """
    Reads images contained in an x-by-2 DataFrame (where column 0 is the
    image path, column 1 is the class name, and x is the number of
    images).

    Arguments:
        df (pandas-DataFrame): x-by-2 DataFrame (where column 0 is the
        image path, column 1 is the class name, and x is the number of
        images)

    Keyword Arguments:
        class_name (str) optional: name of a class in the DataFrame. If
        given, only images belonging to that class will be read. Defaults
        to None.
        -
        save (bool) optional: whether or not to save resulting array of
        image data as a .npz file. Note that saving will create a new
        directory named "read_img_df" if it does not already exist.
        If it does exist, no files inside it will be overwritten.
        Defaults to False.

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
        img = imageio.imread(j[0])
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
        if os.path.exists("preprocessing"):
            if os.path.exists("preprocessing/read_img_df"):
                new_n = len(os.listdir("preprocessing/read_img_df")) + 1
                np.savez(
                    f"preprocessing/read_img_df/X_data_{new_n}",
                    img_array,
                )
            else:
                os.mkdir("preprocessing/read_img_df")
                new_n = len(os.listdir("preprocessing/read_img_df")) + 1
                np.savez(
                    f"preprocessing/read_img_df/X_data_{new_n}",
                    img_array,
                )
        else:
            os.mkdir("preprocessing")
            os.mkdir("preprocessing/read_img_df")
            new_n = len(os.listdir("preprocessing/read_img_df")) + 1
            np.savez(
                f"preprocessing/read_img_df/X_data_{new_n}", img_array
            )

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
        one-hot encoded data as a .npz file. Note that saving will create
        new directory named "one_hot_encoding" if it does not already
        exist. If it does exist, no files inside it will be overwritten.
        Defaults to False.

    Returns:
        y_data (numpy-array): one-hot encoded class label data as numpy-
        array.
    """

    y_list = []
    labels = []
    if type(class_list) is not list:
        raise Exception(
            "class_list must be a list;" + f" {type(class_list)} given."
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
        if os.path.exists("preprocessing"):
            if os.path.exists("preprocessing/one_hot_encoding"):
                new_n = (
                    len(os.listdir("preprocessing/one_hot_encoding"))
                    + 1
                )
                np.savez(
                    f"preprocessing/one_hot_encoding/y_data_{new_n}",
                    y_data,
                )
            else:
                os.mkdir("preprocessing/one_hot_encoding")
                new_n = (
                    len(os.listdir("preprocessing/one_hot_encoding"))
                    + 1
                )
                np.savez(
                    f"preprocessing/one_hot_encoding/y_data_{new_n}",
                    y_data,
                )
        else:
            os.mkdir("preprocessing")
            os.mkdir("preprocessing/one_hot_encoding")
            new_n = (
                len(os.listdir("preprocessing/one_hot_encoding")) + 1
            )
            np.savez(
                f"preprocessing/one_hot_encoding/y_data_{new_n}", y_data
            )

    return y_data


# ------------------------------------------------------------------------


class Image_Dataset:

    """
    Class representing an image dataset.

    Attributes
    ----------
    base_path (str): path to the directory containing images or class
    subdirectories.
    -
    class_list (list): list of classes in dataset.
    -
    class_no (int): number of classes in dataset.
    -
    size (int): number of data points in dataset.
    -
    X_data (numpy-array): image data arrays.
    -
    y_data (numpy-array): data label arrays (one-hot-encoded).
    -
    labels (list): class labels as list.
    -
    min_val (float): minimum pixel value for the whole dataset.
    -
    max_val (float): maximum pixel value for the whole dataset.
    -
    resize (list or tuple) optional/dependent on init: output dimensions
    (width,height) of resized images.
    -
    from_np (list or tuple) optional/dependent on init: paths for X and
    y data arrays if in np form rather than image directory form.
    -
    X_np (numpy-array) optional/dependent on init: image data arrays if in
    np form rather than image directory form.
    -
    y_np (numpy-array) optional/dependent on init: data label arrays if in
    np form rather than image directory form.
    -
    np_classes (list) optional/dependent on init: list of classes if in
    np form rather than image directory form.
    -
    np_prenorm (bool) optional/dependent on init: whether or not the data
    was normalized before initializing the Image_Dataset object.
    -
    df (pandas-DataFrame) optional/dependent on init: dataset as pandas-
    DataFrame if in image directory form.
    -
    split (bool) optional/dependent on tvt_split: whether or not the data-
    set has been split into train/val/testing subsets.
    -
    X_train (numpy-array) optional/dependent on tvt_split: training image
    data arrays if split using tvt_split.
    -
    X_val (numpy-array) optional/dependent on tvt_split: validation image
    data arrays if split using tvt_split.
    -
    X_test (numpy-array) optional/dependent on tvt_split: testing image
    data arrays if split using tvt_split.
    -
    y_train (numpy-array) optional/dependent on tvt_split: training label
    data arrays if split using tvt_split.
    -
    y_val (numpy-array) optional/dependent on tvt_split: validation label
    data arrays if split using tvt_split.
    -
    y_test (numpy-array) optional/dependent on tvt_split: testing label
    data arrays if split using tvt_split.
    -
    train_dict (dict) optional/dependent on tvt_split: names and arrays
    of training and validation data if split using tvt_split.
    -
    test_dict (dict) optional/dependent on tvt_split: names and arrays of
    testing data if split using tvt_split.


    Methods
    -------
    init: constructs all the necessary attributes for the dataset.
    -
    generate: generates X and y data arrays.
    -
    labels: generates y data as list.
    -
    details: prints or displays summary details about the dataset.
    -
    display_batch: displays random batch of images from the dataset.
    -
    tvt_split: splits X and y data into training, validation and testing
    subsets.
    -
    save_as_np: saves dataset (or subsets) as numpy_arrays.
    -
    save_as_imgdirs: saves dataset into main directory and subdirectories
    for each class.
    -
    augment_training_set: calls on an (initialized) imgo.augtools
    augmenter to apply image augmentation to the Image_Dataset's X_train
    subset.
    """

    def __init__(
        self,
        base_path,
        resize=None,
        normalize=False,
        from_np=None,
        np_classes=None,
        np_prenorm=False,
    ):

        """
        Constructs all the necessary attributes for the dataset.

        The data is generated from one of two possible sources. Either
        from a main directory (base path) containing subdirectories for
        each class (inferred) in which the images for each class are
        located, or from numpy-arrays (in .npy or .npz form; inferred).
        The default is the former case, and the latter case is specified
        using the from_np and np_classes arguments.

        Arguments:
            base_path (str): path to the directory containing images or
            class subdirectories.

        Keyword Arguments:
            resize (list or tuple) optional: output dimensions (width,
            height) for resizing images. If None, no resizing will occur.
            Defaults to None.
            -
            normalize (bool) optional: whether or not to normalize image
            pixel values to range [0,1]. Defaults to False.
            -
            from_np (list or tuple) optional: name (with path) for X
            (first) and y (second) data if using numpy-array data.
            Defaults to None. Note that the data must be located within
            the directory specified in the base_path argument, and that
            the y data must be one-hot encoded.
            -
            np_classes (list) optional: list of class names if using
            numpy-array data.
            -
            np_prenorm (bool) optional: whether or not the numpy data has
            been normalized prior to initialization of the Image_Dataset
            object. If it has been normalized, not setting this argument
            to True will result in error. Defaults to False.

        Returns:
            Image_Dataset object.
        """

        self.split = False
        self.base_path = base_path

        if resize is None:
            self.resize = None
        elif (
            (type(resize) is tuple) or (type(resize) is list)
        ) and len(resize) == 2:
            self.resize = resize
        else:
            raise Exception(
                "Resize argument should be list"
                + " or tuple of length 2."
            )
            self.resize = None

        if from_np is None:
            if np_classes is not None:
                raise Exception(
                    "Cannot give np_labels argument"
                    + " if not giving from_np argument."
                )
            self.from_np = None
            self.X_np = None
            self.y_np = None
        elif (
            (type(from_np) is tuple) or (type(from_np) is list)
        ) and len(from_np) == 2:
            self.from_np = from_np
            if self.from_np[0].endswith("npz"):
                npz_X = np.load(
                    f"{base_path}/{from_np[0]}", allow_pickle=True
                )
                self.X_np = npz_X[npz_X.files[0]]
            elif self.from_np[0].endswith("npy"):
                self.X_np = np.load(
                    f"{base_path}/{from_np[0]}", allow_pickle=True
                )
            else:
                raise Exception(
                    "Generation from np arrays requires"
                    + " filetype .npy or .npz"
                )
            if self.from_np[1].endswith("npz"):
                npz_y = np.load(
                    f"{base_path}/{from_np[1]}", allow_pickle=True
                )
                self.y_np = npz_y[npz_y.files[0]]
            elif self.from_np[0].endswith("npy"):
                self.y_np = np.load(
                    f"{base_path}/{from_np[1]}", allow_pickle=True
                )
            else:
                raise Exception(
                    "Generation from np arrays requires"
                    + " filetype .npy or .npz"
                )
            if np_classes is not None:
                if type(np_classes) is list:
                    if len(np_classes) == self.y_np.shape[1]:
                        self.np_classes = sorted(
                            list(set(np_classes)),
                            key=lambda f: f.lower(),
                        )
                    else:
                        raise Exception(
                            "np_classes must have same"
                            + " number of classes as y array."
                        )
                else:
                    raise Exception(
                        "np_classes must be a list;"
                        + f" {type(np_classes)} given."
                    )
            else:
                self.np_classes = [
                    f"class_{i}" for i in np.arange(self.y_np.shape[1])
                ]
        else:
            raise Exception(
                "from_np must be list or tuple;"
                + f" {type(from_np)} given."
            )
            self.from_np = None
            self.X_np = None
            self.y_np = None
            self.np_classes = None

        X_list = []
        y_list = []
        labels = []
        min_pvs = []
        max_pvs = []

        if self.from_np is None:
            self.class_list = get_class_names(self.base_path)
            if self.class_list == []:
                self.class_list.append("no_class")
            else:
                pass
            self.class_no = len(self.class_list)
            self.df = img_to_df(self.base_path)
            self.size = len(self.df)

            for i, j in tqdm(
                self.df.iterrows(),
                total=self.size,
                desc="Processing data",
            ):
                ohe_init = np.zeros(self.class_no)
                label = j[1]
                labels.append(label)
                ohe_init[self.class_list.index(label)] = 1
                ohe_label = ohe_init
                y_list.append(ohe_label)

                raw_img = imageio.imread(j[0])
                raw_dims = np.max(raw_img.shape)
                if self.resize:
                    if (
                        (type(self.resize) is tuple)
                        or (type(self.resize) is list)
                        and len(self.resize) == 2
                    ):
                        scale = (self.resize[0], self.resize[1])
                    else:
                        raise Exception(
                            "Resize argument should be list"
                            + " or tuple of length 2."
                        )
                    if raw_dims < np.max(scale):
                        scaled_img = cv2.resize(
                            raw_img,
                            scale,
                            interpolation=cv2.INTER_CUBIC,
                        )
                    else:
                        scaled_img = cv2.resize(
                            raw_img, scale, interpolation=cv2.INTER_AREA
                        )
                    img = scaled_img
                else:
                    img = raw_img

                X_list.append(img)
                min_pvs.append(np.min(img))
                max_pvs.append(np.max(img))
        else:
            self.class_list = self.np_classes
            self.class_no = len(self.class_list)
            self.size = self.X_np.shape[0]

            for i in tqdm(
                np.arange(self.X_np.shape[0]),
                total=self.size,
                desc="Processing data",
            ):
                if self.np_classes is not None:
                    label_index = np.argmax(self.y_np[i], axis=0)
                    label = self.np_classes[label_index]
                else:
                    label = np.argmax(self.y_np[i], axis=0)
                labels.append(label)
                y_list.append(self.y_np[i])

                if np_prenorm:
                    raw_img = (self.X_np[i] * 255).astype(np.uint8)
                else:
                    raw_img = self.X_np[i]

                raw_dims = np.max(raw_img.shape)

                if self.resize:
                    if (
                        (type(self.resize) is tuple)
                        or (type(self.resize) is list)
                        and len(self.resize) == 2
                    ):
                        scale = (self.resize[0], self.resize[1])
                    else:
                        raise Exception(
                            "Resize argument should be list"
                            + " or tuple of length 2."
                        )
                    if raw_dims < np.max(scale):
                        scaled_img = cv2.resize(
                            raw_img,
                            scale,
                            interpolation=cv2.INTER_CUBIC,
                        )
                    else:
                        scaled_img = cv2.resize(
                            raw_img, scale, interpolation=cv2.INTER_AREA
                        )
                    img = scaled_img
                else:
                    img = raw_img

                X_list.append(img)
                min_pvs.append(np.min(img))
                max_pvs.append(np.max(img))

        X_data = np.array(X_list)
        self.y_data = np.array(y_list)
        self.labels = labels

        self.normal = False
        if normalize:
            print("Normalizing...")
            self.normal = True
            self.X_data = X_data / 255
            self.min_val = np.min(min_pvs) / 255
            self.max_val = np.max(max_pvs) / 255
        else:
            self.X_data = X_data
            self.min_val = np.min(min_pvs)
            self.max_val = np.max(max_pvs)

        print("Image_Dataset initialized.")

    #     ----------

    def generate(self):

        """
        Generates X and y data arrays from Image_Dataset object.
        """

        return self.X_data, self.y_data

    #     ----------

    def labels(self):

        """
        Generates y data as list from Image_Dataset object.
        """

        return self.labels

    #     ----------

    def details(self, plot=False):

        """
        Prints summary details of Image_Dataset object, or displays the
        details as a visualization if kwarg plot is given as True.
        """

        lab, num = np.unique(self.labels, return_counts=True)

        labels_dict = {}
        for i in np.arange(lab.shape[0]):
            labels_dict[lab[i]] = num[i]

        if self.resize is not None:
            image_size = self.resize
        else:
            image_size = "various"

        val_ranges = {"min": self.min_val, "max": self.max_val}

        ds_dict = {
            "total_images": self.size,
            "images_per_class": labels_dict,
            "image_size": image_size,
            "pixel_values": val_ranges,
        }

        if plot:
            class_label = []
            for i in lab:
                if (type(i) is str) or (type(i) is np.str_):
                    class_label.append(i.replace("_", " ").title())
                else:
                    class_label.append(i)

            data_df = pd.DataFrame(
                list(zip(class_label, num)), columns=["class", "images"]
            )

            df = data_df.sort_values(
                by=["class"], inplace=False, ascending=False
            )

            y_pos = list(range(1, len(df) + 1))

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

            plt.hlines(
                y=y_pos,
                xmin=0,
                xmax=df["images"],
                color="#b0faff",
                alpha=0.65,
                linewidth=10,
            )
            plt.plot(
                df["images"],
                y_pos,
                "o",
                markersize=10,
                color="#00d7e6",
            )
            plt.yticks(y_pos, df["class"])

            ax.set_xlabel(
                "Images",
                fontsize=15,
                fontweight="black",
                color="#333F4B",
            )
            ax.set_ylabel("")
            ax.tick_params(axis="both", which="major", labelsize=12)
            ax.spines["top"].set_color("none")
            ax.spines["right"].set_color("none")
            ax.spines["left"].set_smart_bounds(True)
            ax.spines["bottom"].set_smart_bounds(True)
            ax.spines["bottom"].set_position(("axes", -0.04))
            ax.spines["left"].set_position(("axes", 0))

            plt.show()

        else:
            print("Image_Dataset details")
            print("---------------------")
            print(
                "\n".join(
                    "{}: {}".format(k, v) for k, v in ds_dict.items()
                )
            )

    #     ----------

    def display_batch(self, n_rows, n_cols):

        """
        Displays random batch of images from the Image_Dataset object.

        Arguments:
            n_rows (int): number of rows of images to display.
            -
            n_cols (int): number of columns of images to display.

        Returns:
            Visualization of random batch of images from the dataset.
        """

        if n_rows * n_cols > self.size:
            raise Exception(
                f"Cannot display {n_rows*n_cols} images"
                + f" because only {self.size} in dataset."
            )
        else:
            ds_array = np.arange(self.size)
            np.random.shuffle(ds_array)
            index_list = ds_array[0 : n_rows * n_cols]

        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = "Helvetica"
        plt.rcParams["text.color"] = "#333F4B"

        fig = plt.figure(figsize=(12, 8))

        for i in range(1, (n_rows * n_cols) + 1):
            ax = fig.add_subplot(n_rows, n_cols, i)
            if self.normal == True:
                img = (self.X_data[index_list[i - 1]] * 255).astype(
                    np.uint8
                )
            else:
                img = self.X_data[index_list[i - 1]]
            ax.imshow(img)
            ax.set_title(self.labels[index_list[i - 1]])
            ax.set_xticks([])
            ax.set_yticks([])

        fig.tight_layout()
        plt.show()

    #     ----------

    def tvt_split(
        self, splits, seed=None, stratify=False, standardize=False
    ):

        """
        Splits the Image_Dataset object into training, validation, and
        testing subsets. Whether or not to include a validation split is
        inferred by the number of elements in the splits argument.

        Note that this method is built using Scikit-Learn's
        train_test_split. For more information see:
        https://scikit-learn.org/stable/modules/classes.html#

        Arguments:
            splits (list or tuple): ratios used to split the dataset.
            The total must sum to 1. If the argument consists of 2
            elements, the dataset will be split into training and testing
            subsets. If the argument consists of 3 elements, the dataset
            will be split into training, validation, and testing subsets.

        Keyword Arguments:
            seed (int) optional: random seed for use in the data split.
            Defaults to None.
            -
            stratify (bool) optional: whether or not to preserve the class
            balances that exist in the un-split data. Defaults to False.
            -
            standardize (bool) optional: whether or not to standardize
            the pixel values in the training and testing (and validation)
            sets using the mean and standard deviation of the training
            data. Defaults to False.

        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test (numpy-array):
            training, validation, and testing subsets if 3 elements are
            given in the split argument.
            -
            X_train, X_test, y_train, y_test (numpy-array):training and
            testing subsets if 2 elements are given in the split
            argument.
        """

        if self.resize is None:
            raise Exception("Data must be resized in order to split.")

        if ((type(splits) is tuple) or (type(splits) is list)) and sum(
            splits
        ) == 1:
            if len(splits) == 3:
                self.split = True
                val_set = True
                if stratify:
                    (Xtr, Xtv, ytr, ytv) = train_test_split(
                        self.X_data,
                        self.y_data,
                        test_size=round(splits[1] + splits[2], 1),
                        stratify=self.y_data,
                        random_state=seed,
                    )
                    (Xv, Xt, yv, yt) = train_test_split(
                        Xtv,
                        ytv,
                        test_size=splits[2]
                        / round(splits[1] + splits[2], 1),
                        stratify=ytv,
                        random_state=seed,
                    )
                else:
                    (Xtr, Xtv, ytr, ytv) = train_test_split(
                        self.X_data,
                        self.y_data,
                        test_size=round(splits[1] + splits[2], 1),
                        random_state=seed,
                    )
                    (Xv, Xt, yv, yt) = train_test_split(
                        Xtv,
                        ytv,
                        test_size=splits[2]
                        / round(splits[1] + splits[2], 1),
                        random_state=seed,
                    )

                self.y_train = ytr
                self.y_val = yv
                self.y_test = yt

            elif len(splits) == 2:
                self.split = True
                val_set = False
                if stratify:
                    (Xtr, Xt, ytr, yt) = train_test_split(
                        self.X_data,
                        self.y_data,
                        test_size=splits[1],
                        stratify=self.y_data,
                        random_state=seed,
                    )
                else:
                    (Xtr, Xt, ytr, yt) = train_test_split(
                        self.X_data,
                        self.y_data,
                        test_size=splits[1],
                        random_state=seed,
                    )

                self.y_train = ytr
                self.y_val = None
                self.y_test = yt

            else:
                self.split = False
                raise Exception(
                    "Splits argument must be list or tuple"
                    + " of either 2 or 3 elements."
                )

            if standardize:
                if self.resize is None:
                    raise Exception(
                        "Images must all have the same dimensions in"
                        + " order to be standardized."
                    )
                else:
                    self.split_std = True
                    mu = np.mean(Xtr)
                    sigma = np.std(Xtr)

                    zXtr = (Xtr - mu) / sigma
                    self.X_train = np.exp(zXtr) / (1 + np.exp(zXtr))
                    zXt = (Xt - mu) / sigma
                    self.X_test = np.exp(zXt) / (1 + np.exp(zXt))

                    if val_set == True:
                        zXv = (Xv - mu) / sigma
                        self.X_val = np.exp(zXv) / (1 + np.exp(zXv))
                    else:
                        self.X_val = None

            else:
                self.split_std = False
                self.X_train = Xtr
                self.X_test = Xt
                if val_set == True:
                    self.X_val = Xv
                else:
                    self.X_val = None

            print("X_train shape:", self.X_train.shape)
            print("y_train shape:", self.y_train.shape)
            if self.X_val is not None:
                print("X_val shape:", self.X_val.shape)
                print("y_val shape:", self.y_val.shape)
                self.train_dict = {
                    "X_train": self.X_train,
                    "X_val": self.X_val,
                    "y_train": self.y_train,
                    "y_val": self.y_val,
                }
            else:
                self.train_dict = {
                    "X_train": self.X_train,
                    "y_train": self.y_train,
                }
            print("X_test shape:", self.X_test.shape)
            print("y_test shape:", self.y_test.shape)

            self.test_dict = {
                "X_test": self.X_test,
                "y_test": self.y_test,
            }

        else:
            raise Exception(
                "Splits argument must be list or tuple"
                + " of either 2 or 3 elements which sum to 1."
            )

    #     ----------

    def save_as_np(
        self, save_path, save_mode, save_split=False, overwrite=False
    ):

        """
        Saves the Image_Dataset object as X and y numpy-arrays. Creates
        a new directory for each subset saved (if split), or a directory
        for X and y data arrays (if not split).

        Arguments:
            save_path (str): path to directory in which the data is to
            be saved.
            -
            save_mode (str: 'npy' or 'npz') optional: whether to save
            the data arrays in .npy or .npz format.

        Keyword Arguments:
            save_split (bool) optional: if the Image_Dataset object
            has been split using tvt_split, whether or not to save
            each subset separately. Will result in error if the dataset
            has not been split. Defaults to False.
            -
            overwrite (bool) optional: whether or not to overwrite existing
            directories when saving. If False, arrays will be saved with
            timestamp if other arrays already exist in the save location.
            Defaults to False.

        Returns:
            Successful save message.
        """

        if save_mode not in ["npz", "npy"]:
            raise Exception(
                "Must select valid save mode:"
                + ' either "npz" or "npy".'
            )
        else:
            ext = save_mode

        if save_split:
            if self.split == False:
                raise Exception("Data set has not been split.")
            else:
                if os.path.exists(f"{save_path}/train_data"):
                    if save_mode == "npz":
                        if overwrite:
                            send2trash(f"{save_path}/train_data")
                            os.mkdir(f"{save_path}/train_data")
                            for k, v in self.train_dict.items():
                                np.savez(
                                    f"{save_path}/train_data/{k}", v
                                )
                                print(
                                    f"{k}.{ext} saved in {save_path}/train_data/"
                                )
                        else:
                            for k, v in self.train_dict.items():
                                np.savez(
                                    f"{save_path}/train_data/{k}_"
                                    + f"{str(datetime.now())[:19].replace(' ','_').replace('-','_').replace(':','_')}",
                                    v,
                                )
                                print(
                                    f"{k}.{ext} saved in {save_path}/train_data/"
                                )
                    else:
                        if overwrite:
                            send2trash(f"{save_path}/train_data")
                            os.mkdir(f"{save_path}/train_data")
                            for k, v in self.train_dict.items():
                                np.save(
                                    f"{save_path}/train_data/{k}", v
                                )
                                print(
                                    f"{k}.{ext} saved in {save_path}/train_data/"
                                )
                        else:
                            for k, v in self.train_dict.items():
                                np.save(
                                    f"{save_path}/train_data/{k}_"
                                    + f"{str(datetime.now())[:19].replace(' ','_').replace('-','_').replace(':','_')}",
                                    v,
                                )
                                print(
                                    f"{k}.{ext} saved in {save_path}/train_data/"
                                )
                else:
                    if os.path.exists(f"{save_path}"):
                        os.mkdir(f"{save_path}/train_data")
                    else:
                        os.mkdir(f"{save_path}")
                        os.mkdir(f"{save_path}/train_data")

                    if save_mode == "npz":
                        for k, v in self.train_dict.items():
                            np.savez(f"{save_path}/train_data/{k}", v)
                            print(
                                f"{k}.{ext} saved in {save_path}/train_data/"
                            )
                    else:
                        for k, v in self.train_dict.items():
                            np.save(f"{save_path}/train_data/{k}", v)
                            print(
                                f"{k}.{ext} saved in {save_path}/train_data/"
                            )

                if os.path.exists(f"{save_path}/test_data"):
                    if save_mode == "npz":
                        if overwrite:
                            send2trash(f"{save_path}/test_data")
                            os.mkdir(f"{save_path}/test_data")
                            for k, v in self.test_dict.items():
                                np.savez(
                                    f"{save_path}/test_data/{k}", v
                                )
                                print(
                                    f"{k}.{ext} saved in {save_path}/test_data/"
                                )
                        else:
                            for k, v in self.test_dict.items():
                                np.savez(
                                    f"{save_path}/test_data/{k}_"
                                    + f"{str(datetime.now())[:19].replace(' ','_').replace('-','_').replace(':','_')}",
                                    v,
                                )
                                print(
                                    f"{k}.{ext} saved in {save_path}/test_data/"
                                )
                    if save_mode == "npy":
                        if overwrite:
                            send2trash(f"{save_path}/test_data")
                            os.mkdir(f"{save_path}/test_data")
                            for k, v in self.test_dict.items():
                                np.save(f"{save_path}/test_data/{k}", v)
                                print(
                                    f"{k}.{ext} saved in {save_path}/test_data/"
                                )
                        else:
                            for k, v in self.test_dict.items():
                                np.save(
                                    f"{save_path}/test_data/{k}_"
                                    + f"{str(datetime.now())[:19].replace(' ','_').replace('-','_').replace(':','_')}",
                                    v,
                                )
                                print(
                                    f"{k}.{ext} saved in {save_path}/test_data/"
                                )
                else:
                    if os.path.exists(f"{save_path}"):
                        os.mkdir(f"{save_path}/test_data")
                    else:
                        os.mkdir(f"{save_path}")
                        os.mkdir(f"{save_path}/test_data")

                    if save_mode == "npz":
                        for k, v in self.test_dict.items():
                            np.savez(f"{save_path}/test_data/{k}", v)
                            print(
                                f"{k}.{ext} saved in {save_path}/test_data/"
                            )
                    else:
                        for k, v in self.test_dict.items():
                            np.save(f"{save_path}/test_data/{k}", v)
                            print(
                                f"{k}.{ext} saved in {save_path}/test_data/"
                            )

        else:
            if os.path.exists(f"{save_path}"):
                if overwrite:
                    if os.path.exists(f"{save_path}/X_data"):
                        send2trash(f"{save_path}/X_data")
                    if os.path.exists(f"{save_path}/y_data"):
                        send2trash(f"{save_path}/y_data")
                if save_mode == "npz":
                    np.savez(f"{save_path}/X_data", self.X_data)
                    print(f"X_data.{ext} saved in {save_path}/")
                    np.savez(f"{save_path}/y_data", self.y_data)
                    print(f"y_data.{ext} saved in {save_path}/")
                else:
                    np.save(f"{save_path}/X_data", self.X_data)
                    print(f"X_data.{ext} saved in {save_path}/")
                    np.save(f"{save_path}/y_data", self.y_data)
                    print(f"y_data.{ext} saved in {save_path}/")
            else:
                os.mkdir(f"{save_path}")
                if save_mode == "npz":
                    np.savez(f"{save_path}/X_data", self.X_data)
                    print(f"X_data.{ext} saved in {save_path}/")
                    np.savez(f"{save_path}/y_data", self.y_data)
                    print(f"y_data.{ext} saved in {save_path}/")
                else:
                    np.save(f"{save_path}/X_data", self.X_data)
                    print(f"X_data.{ext} saved in {save_path}/")
                    np.save(f"{save_path}/y_data", self.y_data)
                    print(f"y_data.{ext} saved in {save_path}/")

    #     ----------

    def save_as_imgdirs(self, save_path, overwrite=False):

        """
        Saves the Image_Dataset object as images in a main directory with
        subdirectories for each class. Creates a new subdirectory named
        ds_images inside the main diretory (specified using the save_path
        argument) in which each class subdirectory will be saved.

        Arguments:
            save_path (str): path to directory in which the data is to
            be saved.

        Keyword Arguments:
            overwrite (bool) optional: whether or not to overwrite existing
            directories when saving. If False, images will be saved with
            timestamp if other images already exist in the save location.
            Defaults to False.

        Returns:
            Successful save message.
        """

        if os.path.exists(f"{save_path}"):
            if os.path.exists(f"{save_path}/ds_images"):
                for i in self.class_list:
                    if os.path.exists(f"{save_path}/ds_images/{i}"):
                        if overwrite:
                            send2trash(f"{save_path}/ds_images/{i}")
                            os.mkdir(f"{save_path}/ds_images/{i}")
                        else:
                            pass
                    else:
                        os.mkdir(f"{save_path}/ds_images/{i}")
            else:
                os.mkdir(f"{save_path}/ds_images")
                for i in self.class_list:
                    if os.path.exists(f"{save_path}/ds_images/{i}"):
                        if overwrite:
                            send2trash(f"{save_path}/ds_images/{i}")
                            os.mkdir(f"{save_path}/ds_images/{i}")
                        else:
                            pass
                    else:
                        os.mkdir(f"{save_path}/ds_images/{i}")
        else:
            os.mkdir(f"{save_path}")
            os.mkdir(f"{save_path}/ds_images")
            for i in self.class_list:
                if os.path.exists(f"{save_path}/ds_images/{i}"):
                    pass
                else:
                    os.mkdir(f"{save_path}/ds_images/{i}")

        list_by_class = []
        for c in self.class_list:
            class_instances = []
            for y in np.arange(self.y_data.shape[0]):
                if self.class_list[np.argmax(self.y_data[y])] == c:
                    class_instances.append(self.X_data[y])
            list_by_class.append(np.array(class_instances))
        for n, l in enumerate(list_by_class):
            for i, j in tqdm(enumerate(l), total=len(l)):
                if self.normal == True:
                    img = (j * 255).astype(np.uint8)
                else:
                    img = j
                if os.path.exists(
                    f"{save_path}/ds_images/"
                    + f"{self.class_list[n]}"
                    + f"/{self.class_list[n]}_{i+1}.jpg"
                ):
                    if overwrite:
                        imageio.imwrite(
                            f"{save_path}/ds_images/"
                            + f"{self.class_list[n]}"
                            + f"/{self.class_list[n]}_{i+1}.jpg",
                            img,
                        )
                    else:
                        imageio.imwrite(
                            f"{save_path}/ds_images/"
                            + f"{self.class_list[n]}"
                            + f"/{self.class_list[n]}_{i+1}"
                            + f"_{str(datetime.now())[:19].replace(' ','_').replace('-','_').replace(':','_')}.jpg",
                            img,
                        )
                else:
                    imageio.imwrite(
                        f"{save_path}/ds_images/"
                        + f"{self.class_list[n]}"
                        + f"/{self.class_list[n]}_{i+1}.jpg",
                        img,
                    )

        print(f"Images saved in {save_path}/ds_images/")

    #     ----------

    def augment_training_set(self, augmenter, portion, inplace=False):

        """
        Calls on an (initialized) imgo.augtools augmenter to apply image
        augmentation to the Image_Dataset's X_train subset.

        Arguments:
            augmenter (imgo.uptools Augmenter object): the augmenter to
            apply to the images.
            -
            portion (float): float within the range [0,1]. This is the
            portion of images in the set that will be augmented.

        Keyword Arguments:
            inplace (bool) optional: whether or not to replace the images
            in the Image_Dataset's X_train subset with the augmented
            images. If True, the X_train attribute will be changed. If
            False, will return a new numpy-array featuring the augmented
            images.

        Returns:
            X_train_aug (numpy-array): the Image_Dataset's X_train object
            with augmented images (if inplace argument set to False).
        """

        from imgo import augtools

        if self.split == True:

            if (portion >= 0) and (portion <= 1):
                n = np.round(self.X_train.shape[0] * (portion)).astype(
                    np.uint8
                )
                img_indices = np.random.choice(
                    self.X_train.shape[0],
                    np.min([self.X_train.shape[0], n]),
                    replace=False,
                )
            else:
                raise Exception(
                    "Portion argument must be in range [0,1]."
                )

            X_train_aug = []
            for x in tqdm(np.arange(self.X_train.shape[0])):
                if x in img_indices:
                    if (self.normal == True) or (
                        self.split_std == True
                    ):
                        aug_img = augmenter.random_augment(
                            self.X_train[x], prenorm=True
                        )
                    else:
                        aug_img = augmenter.random_augment(
                            self.X_train[x], prenorm=False
                        )
                    X_train_aug.append(aug_img)
                else:
                    X_train_aug.append(self.X_train[x])

            if inplace:
                self.X_train = np.array(X_train_aug)
            else:
                return np.array(X_train_aug)

        else:
            raise Exception("Data has not been split.")
