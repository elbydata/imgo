"""
IMGO - Process, augment, and balance image data.
------------------------------------------------
AUGTOOLS module: 

Last updated: version 2.5.4

Classes
-------        
Augmenter: Class representing an image augmentation system.

    Class Attributes:
        rotate_range (list or tuple): range in degrees from which a random 
        rotation angle is drawn.
        -
        shear_range (list or tuple): range in degrees from which a random 
        shear angle is drawn.
        -
        dropout_pair (list or tuple): amount and degree of random pixel 
        dropout applied.
        -
        x_scale (list or tuple): range relative to image size from which 
        a random x-axis rescale value is drawn.
        -
        y_scale (list or tuple): range relative to image size from which 
        a random y-axis rescale value is drawn.
        -
        x_shift (list or tuple): range relative to image size from which 
        a random x-axis translation is drawn.
        -
        y_shift (list or tuple): range relative to image size from which 
        a random y-axis translation is drawn.
        -
        clip_limit (list or tuple): range from which random clip limits 
        for CLAH equalization are drawn.
        -
        pwa_scale (list or tuple): range relative to image size from which 
        random piecewise affine translations are drawn.
        -
        h_flip (float): proportion of images on which a horizontal flip is 
        performed.
        -
        v_flip (float): proportion of images on which a vertical flip is 
        performed.
        -
        g_sev (int): amount (or range if batch is randomized) of severity 
        with which gaussian noise is applied.
        -
        b_sev (int): amount (or range if batch is randomized) of severity 
        with which brightness is altered.
        -
        e_sev (int): amount (or range if batch is randomized) of severity 
        with which elastic transformation is applied.
        -
        contrast (bool): whether or not random contrast adjustment is 
        applied.
        -
        sharpness (bool): whether or not random sharpness adjustment is 
        applied.
        -
        fill_mode (str): which method is used to fill pixels created by 
        any of the augmentation functions.
        -
        randomize_params (bool): whether the augmentation parameters 
        applied are fixed as the input given or are randomly drawn from a 
        range inferred from the input given.
        -
        f_list (list): list of augmentation functions that have been 
        applied.
        -
        argno (int): number of augmentation functions that have been 
        applied.   
    
    Class Methods:
        init: constructs all the necessary attributes for the augmenter.
        -
        summary: display summary information about the augmenter.
        -
        aug_rotate: apply imgaug Affine rotation to image.
        -
        aug_shear: apply imgaug Affine shear to image.
        -
        aug_dropout: apply imgaug CoarseDropout to image.
        -
        aug_x_scale: apply imgaug ScaleX to image.
        -
        aug_y_scale: apply imgaug ScaleY to image.
        -
        aug_x_shift: apply imgaug TranslateX to image.
        -
        aug_y_shift: apply imgaug TranslateY to image.
        -
        aug_clahe: apply imgaug CLAHE to image.
        -
        aug_pwa: apply imgaug PiecewiseAffine to image.
        -
        aug_h_flip: apply imgaug Fliplr to image.
        -
        aug_v_flip: apply imgaug Flipud to image.
        -
        aug_g_noise: apply imgaug GaussianNoise to image.
        -
        aug_brightness: apply imgaug Brightness to image.
        -
        aug_elastic: apply imgaug ElasticTransform to image.
        -
        aug_contrast: apply imgaug EnhanceContrast to image.
        -
        aug_sharpness: apply imgaug EnhanceSharpness to image.
        -
        simple_augment: apply one or more augmentations to an image.
        -
        random_augment: randomly apply one or more augmentations to image.
        -
        display: augment and display an input image.
        -
        display_sample: augment and display a sample image from a 
        collection.
        -
        augment_flow: apply augmentation to images located in directories 
        on disk, and save augmented images to disk.
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


class Augmenter:

    """
    Class representing an image augmentation system.

    Note that the augmentation system is built using imgaug and offers
    a smaller range of augmentation options than imgaug. Additional
    augmentation functions can be added herein if required.
    For more information see:
    https://imgaug.readthedocs.io/en/latest/

    Attributes
    ----------
    rotate_range (list or tuple): range in degrees from which a random
    rotation angle is drawn.
    -
    shear_range (list or tuple): range in degrees from which a random
    shear angle is drawn.
    -
    dropout_pair (list or tuple): amount and degree of random pixel
    dropout applied.
    -
    x_scale (list or tuple): range relative to image size from which a
    random x-axis rescale value is drawn.
    -
    y_scale (list or tuple): range relative to image size from which a
    random y-axis rescale value is drawn.
    -
    x_shift (list or tuple): range relative to image size from which a
    random x-axis translation is drawn.
    -
    y_shift (list or tuple): range relative to image size from which a
    random y-axis translation is drawn.
    -
    clip_limit (list or tuple): range from which random clip limits for
    CLAH equalization are drawn.
    -
    pwa_scale (list or tuple): range relative to image size from which
    random piecewise affine translations are drawn.
    -
    h_flip (float): proportion of images on which a horizontal flip is
    performed.
    -
    v_flip (float): proportion of images on which a vertical flip is
    performed.
    -
    g_sev (int): amount (or range if batch is randomized) of severity with
    which gaussian noise is applied.
    -
    b_sev (int): amount (or range if batch is randomized) of severity with
    which brightness is altered.
    -
    e_sev (int): amount (or range if batch is randomized) of severity with
    which elastic transformation is applied.
    -
    contrast (bool): whether or not random contrast adjustment is applied.
    -
    sharpness (bool): whether or not random sharpness adjustment is
    applied.
    -
    fill_mode (str): which method is used to fill pixels created by any of
    the augmentation functions.
    -
    randomize_params (bool): whether the augmentation parameters applied
    are fixed as the input given or are randomly drawn from a range
    inferred from the input given.
    -
    f_list (list): list of augmentation functions that have been applied.
    -
    argno (int): number of augmentation functions that have been applied.

    Methods
    -------
    init: constructs all the necessary attributes for the augmenter.
    -
    summary: display summary information about the augmenter.
    -
    aug_rotate: apply imgaug Affine rotation to image.
    -
    aug_shear: apply imgaug Affine shear to image.
    -
    aug_dropout: apply imgaug CoarseDropout to image.
    -
    aug_x_scale: apply imgaug ScaleX to image.
    -
    aug_y_scale: apply imgaug ScaleY to image.
    -
    aug_x_shift: apply imgaug TranslateX to image.
    -
    aug_y_shift: apply imgaug TranslateY to image.
    -
    aug_clahe: apply imgaug CLAHE to image.
    -
    aug_pwa: apply imgaug PiecewiseAffine to image.
    -
    aug_h_flip: apply imgaug Fliplr to image.
    -
    aug_v_flip: apply imgaug Flipud to image.
    -
    aug_g_noise: apply imgaug GaussianNoise to image.
    -
    aug_brightness: apply imgaug Brightness to image.
    -
    aug_elastic: apply imgaug ElasticTransform to image.
    -
    aug_contrast: apply imgaug EnhanceContrast to image.
    -
    aug_sharpness: apply imgaug EnhanceSharpness to image.
    -
    simple_augment: apply one or more augmentations to an image.
    -
    random_augment: randomly apply one or more augmentations to image.
    -
    display: augment and display an input image.
    -
    display_sample: augment and display a sample image from a collection.
    -
    augment_flow: apply augmentation to images located in directories
    on disk, and save augmented images to disk.
    """

    def __init__(
        self,
        rotate_range=None,
        shear_range=None,
        dropout_pair=None,
        x_scale=None,
        y_scale=None,
        x_shift=None,
        y_shift=None,
        clip_limit=None,
        pwa_scale=None,
        h_flip=None,
        v_flip=None,
        g_sev=None,
        b_sev=None,
        e_sev=None,
        contrast=False,
        sharpness=False,
        fill_mode=None,
        randomize_params=False,
    ):

        """
        Constructs all the necessary attributes for the augmenter.

        Once initialized, the augmenter applies one or more augmentation
        functions to the image, depending on which keyword arguments are
        given, in a random order. If the batch is randomized (specified
        with the randomize_params argument), the functions will be applied
        using a random parameter inferred from the argument given.

        Note that in order to be augmented, images must be in non-
        normalized.

        Keyword Arguments:
            rotate_range (list or tuple) optional: range (min, max) in
            degrees from which a random rotation angle is drawn. The
            corresponding aug_rotate function will rotate each image by
            a randomly chosen angle within this range. Defaults to None.
            -
            shear_range (list or tuple) optional: range (min, max) in
            degrees from which a random shear angle is drawn. The
            corresponding aug_shear function will apply a shear
            adjustment to each image at a randomly chosen angle within
            this range. Defaults to None.
            -
            dropout_pair (list or tuple) optional: amount and
            degree (or ranges of both if batch is randomized) of random
            pixel dropout applied, where the first element in the list
            or tuple is the proportion of pixels to be dropped, and the
            second is the proportional size of the image from which to
            drop the pixels (thereby increasing the size of the dropped
            portions). Defaults to None.
            -
            x_scale (list or tuple) optional: range (min, max)
            proportional to image size from which an x-axis scale value is
            drawn. The corresponding aug_x_scale function will scale the
            image to this value along the x-axis. Defaults to None.
            -
            y_scale (list or tuple) optional: range (min, max)
            proportional to image size from which an y-axis scale value is
            drawn. The corresponding aug_y_scale function will scale the
            image to this value along the y-axis. Defaults to None.
            -
            x_shift (list or tuple) optional: range relative
            to image size from which a random x-axis translation is drawn.
            The corresponding aug_x_shift function will shift each image
            on the x-axis by a randomly chosen proportion within this
            range. Defaults to None.
            -
            y_shift (list or tuple) optional: range relative
            to image size from which a random y-axis translation is drawn.
            The corresponding aug_y_shift function will shift each image
            on the y-axis by a randomly chosen proportion within this
            range. Defaults to None.
            -
            clip_limit (list or tuple) optional: range from
            which random clip limits for CLAH equalization are drawn.
            The corresponding aug_clahe function will apply a CLAH
            equalization to each image using a clip limit uniformly
            sampled from the range. Defaults to None.
            -
            pwa_scale (list or tuple) optional: range relative
            to image size from which random piecewise affine translations
            are drawn. The first element of the list or tuple is the
            upper bound of the range from which the translation amount of
            each point in the image is randomly sampled. The second
            element is the upper bound of the range from which the first
            element is sampled, and is randomly applied to each image.
            The corresponding aug_pwa function will apply a piecewise
            affine translation to local areas of the image using these
            ranges. Defaults to None.
            -
            h_flip (float) optional: proportion of images on
            which a horizontal flip is performed. The corresponding
            aug_h_flip function will flip these images horizontally.
            Defaults to None.
            -
            v_flip (float) optional: proportion of images on
            which a vertical flip is performed. The corresponding
            aug_v_flip function will flip these images vertically.
            Defaults to None.
            -
            g_sev (int) optional: amount (or range if batch
            is randomized) of severity with which Gaussian noise is
            applied. The corresponding aug_g_noise function will apply
            noise to the images. Defaults to None.
            -
            b_sev (int) optional: amount (or range if batch
            is randomized) of severity with which brightness is altered.
            The corresponding aug_brightness function will change the
            brightness of the images. Defaults to None.
            -
            e_sev (int) optional: amount (or range if batch
            is randomized) of severity with which elastic transformation
            is applied. The corresponding aug_elastic function will
            apply the transformation to local areas of the image.
            Defaults to None.
            -
            contrast (bool) optional: whether or not random
            contrast adjustment is applied. If the batch is randomized, a
            random proportion of the images will be adjusted by the
            corresponding aug_contrast function. Defaults to False.
            -
            sharpness (bool) optional: whether or not random
            sharpness adjustment is applied. If the batch is randomized,
            a random proportion of the images will be adjusted by the
            corresponding aug_sharpness function. Defaults to False.
            -
            fill_mode (str) optional: which method to use to fill any
            pixels created by any of the augmentation functions. Choose
            from "edge" (fills with the same value as that of the edge of
            the array), or "reflect" (reflects the pixel values along edge
            of the array). If None, will fill with a constant value.
            Defaults to None.
            -
            randomize_params (bool) optional: whether the
            augmentation parameters applied are fixed as the input given
            or are randomly drawn from a range inferred from the input
            given. If False, the same transformation parameters will be
            used for each image in the batch. Defaults to False.

        Returns:
            Augmenter object.
        """

        # Note: if adding augmentation functions, be sure to add to:
        # (1) __init__ arguments
        # (2) setattribute dictionaries
        # (3) the function itself

        for k, v in {
            "rotate_range": rotate_range,
            "shear_range": shear_range,
            "dropout_pair": dropout_pair,
            "x_scale": x_scale,
            "y_scale": y_scale,
            "x_shift": x_shift,
            "y_shift": y_shift,
            "clip_limit": clip_limit,
            "pwa_scale": pwa_scale,
        }.items():

            if v is None:
                setattr(self, k, None)
            elif ((type(v) is tuple) or (type(v) is list)) and len(
                v
            ) == 2:
                setattr(self, k, v)
            else:
                setattr(self, k, None)

        for k, v in {
            "h_flip": h_flip,
            "v_flip": v_flip,
            "g_sev": g_sev,
            "b_sev": b_sev,
            "e_sev": e_sev,
        }.items():

            if (v is None) or (v == 0):
                setattr(self, k, None)
            else:
                setattr(self, k, v)

        for k, v in {
            "contrast": contrast,
            "sharpness": sharpness,
            "randomize_params": randomize_params,
        }.items():

            if v == True:
                setattr(self, k, True)
            else:
                setattr(self, k, False)

        if fill_mode is not None:
            if fill_mode in ["edge", "reflect"]:
                self.fill_mode = fill_mode
            else:
                raise Exception(
                    "Please select valid 'fill_mode': 'edge', or 'reflect'."
                )
                self.fill_mode = "constant"
        else:
            self.fill_mode = "constant"

        f_names = sorted(
            [
                f
                for f in dir(self)
                if callable(getattr(self, f)) and f.startswith("aug_")
            ],
            key=lambda f: f.lower(),
        )
        self.f_list = [getattr(self, i) for i in f_names]

        self.argno = len(self.f_list)

    #     ----------

    def details(self):

        """
        Prints summary of parameters with which the selfmenter was
        initialized.
        """

        exclude_set = set(
            ["f_list", "argno", "randomize_params", "fill_mode"]
        )
        param_list = sorted(
            list(set([k for k in vars(self).keys()]) - exclude_set),
            key=lambda f: f.lower(),
        )
        add_list = ["fill_mode", "randomize_params"]
        print(
            "Augmenter initialized with the following parameter ranges:"
        )
        print(
            "----------------------------------------------------------"
        )
        for i in param_list:
            val = str(vars(self)[i])
            print(f"{i:<20}{val}")
        print("-")
        for i in add_list:
            val = str(vars(self)[i])
            print(f"{i:<20}{val}")

    #     ----------

    def aug_rotate(self, img, pre_norm=False):

        """
        Rotates the image by a random angle within the range given
        (set pre_norm to True if image has been normalized to [0,1]).
        """

        if self.rotate_range is not None:
            rotate = iaa.Affine(
                rotate=self.rotate_range, mode=self.fill_mode
            )
            if pre_norm:
                img = (img * 255).astype(np.uint8)
                rotated_img = (rotate(image=img)) / 255
            else:
                rotated_img = rotate(image=img)
            return rotated_img
        else:
            return img

    #     ----------

    def aug_shear(self, img, pre_norm=False):

        """
        Shear-transforms the image by a random angle within the range
        given (set pre_norm to True if image has been normalized to
        [0,1]).
        """

        if self.shear_range is not None:
            shear = iaa.Affine(
                shear=self.shear_range, mode=self.fill_mode
            )
            if pre_norm:
                img = (img * 255).astype(np.uint8)
                shear_img = (shear(image=img)) / 255
            else:
                shear_img = shear(image=img)
            return shear_img
        else:
            return img

    #     ----------

    def aug_dropout(self, img, pre_norm=False):

        """
        Drops a portion of pixel values to zero (set pre_norm to True
        if image has been normalized to [0,1]).
        """

        if self.dropout_pair is not None:
            if self.randomize_params == True:
                drop_1 = (
                    np.random.randint(0, self.dropout_pair[0] * 100)
                    / 100
                )
                drop_2 = (
                    np.random.randint(self.dropout_pair[1] * 100, 100)
                    / 100
                )
                dropout = iaa.CoarseDropout(drop_1, size_percent=drop_2)
            else:
                dropout = iaa.CoarseDropout(
                    self.dropout_pair[0],
                    size_percent=self.dropout_pair[1],
                )
            if pre_norm:
                img = (img * 255).astype(np.uint8)
                dropout_img = (dropout(image=img)) / 255
            else:
                dropout_img = dropout(image=img)
            return dropout_img
        else:
            return img

    #     ----------

    def aug_x_scale(self, img, pre_norm=False):

        """
        Scales (squashes or stretches) the image on the x-axis by a
        random amount within the range given (set pre_norm to True
        if image has been normalized to [0,1]).
        """

        if self.x_scale is not None:
            x_scale = iaa.ScaleX(self.x_scale, mode=self.fill_mode)
            if pre_norm:
                img = (img * 255).astype(np.uint8)
                x_scale_img = (x_scale(image=img)) / 255
            else:
                x_scale_img = x_scale(image=img)
            return x_scale_img
        else:
            return img

    #     ----------

    def aug_y_scale(self, img, pre_norm=False):

        """
        Scales (squashes or stretches) the image on the y-axis by a
        random amount within the range given (set pre_norm to True
        if image has been normalized to [0,1]).
        """

        if self.y_scale is not None:
            y_scale = iaa.ScaleY(self.y_scale, mode=self.fill_mode)
            if pre_norm:
                img = (img * 255).astype(np.uint8)
                y_scale_img = (y_scale(image=img)) / 255
            else:
                y_scale_img = y_scale(image=img)
            return y_scale_img
        else:
            return img

    #     ----------

    def aug_x_shift(self, img, pre_norm=False):

        """
        Shifts the image on the x-axis by a random amount within the
        range given (set pre_norm to True if image has been
        normalized to [0,1]).
        """

        if self.x_shift is not None:
            x_shift = iaa.TranslateX(
                percent=self.x_shift, mode=self.fill_mode
            )
            if pre_norm:
                img = (img * 255).astype(np.uint8)
                x_shift_img = (x_shift(image=img)) / 255
            else:
                x_shift_img = x_shift(image=img)
            return x_shift_img
        else:
            return img

    #     ----------

    def aug_y_shift(self, img, pre_norm=False):

        """
        Shifts the image on the y-axis by a random amount within the
        range given (set pre_norm to True if image has been
        normalized to [0,1]).
        """

        if self.y_shift is not None:
            y_shift = iaa.TranslateY(
                percent=self.y_shift, mode=self.fill_mode
            )
            if pre_norm:
                img = (img * 255).astype(np.uint8)
                y_shift_img = (y_shift(image=img)) / 255
            else:
                y_shift_img = y_shift(image=img)
            return y_shift_img
        else:
            return img

    #     ----------

    def aug_clahe(self, img, pre_norm=False):

        """
        Applies a CLAH equalization using a random clip limit within
        the range given (set pre_norm to True if image has been
        normalized to [0,1]).
        """

        if self.clip_limit is not None:
            clahe = iaa.CLAHE(clip_limit=self.clip_limit)
            if pre_norm:
                img = (img * 255).astype(np.uint8)
                clahe_img = (clahe(image=img)) / 255
            else:
                clahe_img = clahe(image=img)
            return clahe_img
        else:
            return img

    #     ----------

    def aug_pwa(self, img, pre_norm=False):

        """
        Shifts local pixel areas by a random amount within the range
        given (set pre_norm to True if image has been normalized
        to [0,1]).
        """

        if self.pwa_scale is not None:
            pwa = iaa.PiecewiseAffine(
                scale=self.pwa_scale, mode=self.fill_mode
            )
            if pre_norm:
                img = (img * 255).astype(np.uint8)
                pwa_img = (pwa(image=img)) / 255
            else:
                pwa_img = pwa(image=img)
            return pwa_img
        else:
            return img

    #     ----------

    def aug_h_flip(self, img, pre_norm=False):

        """
        Flips a random subset of images horizontally (set pre_norm
        to True if image has been normalized to [0,1]).
        """

        if self.h_flip is not None:
            h_flip = iaa.Fliplr(self.h_flip)
            if pre_norm:
                img = (img * 255).astype(np.uint8)
                h_flip_img = (h_flip(image=img)) / 255
            else:
                h_flip_img = h_flip(image=img)
            return h_flip_img
        else:
            return img

    #     ----------

    def aug_v_flip(self, img, pre_norm=False):

        """
        Flipts a random subset of images vertically (set pre_norm
        to True if image has been normalized to [0,1]).
        """

        if self.v_flip is not None:
            v_flip = iaa.Flipud(self.v_flip)
            if pre_norm:
                img = (img * 255).astype(np.uint8)
                v_flip_img = (v_flip(image=img)) / 255
            else:
                v_flip_img = v_flip(image=img)
            return v_flip_img
        else:
            return img

    #     ----------

    def aug_g_noise(self, img, pre_norm=False):

        """
        Adds Gaussian noise to the image (set pre_norm to True if
        image has been normalized to [0,1]).
        """

        if self.g_sev is not None:
            rand_sev = np.random.randint(self.g_sev + 1)
            if self.randomize_params == True:
                if rand_sev == 0:
                    g_noise = None
                else:
                    g_noise = iaa.imgcorruptlike.GaussianNoise(
                        severity=rand_sev
                    )
            else:
                g_noise = iaa.imgcorruptlike.GaussianNoise(
                    severity=self.g_sev
                )
            if g_noise is None:
                return img
            else:
                if pre_norm:
                    img = (img * 255).astype(np.uint8)
                    g_noise_img = (g_noise(image=img)) / 255
                else:
                    g_noise_img = g_noise(image=img)
                return g_noise_img
        else:
            return img

    #     ----------

    def aug_brightness(self, img, pre_norm=False):

        """
        Adjusts brightness of the image (set pre_norm to True if
        image has been normalized to [0,1]).
        """

        if self.b_sev is not None:
            rand_sev = np.random.randint(self.b_sev + 1)
            if self.randomize_params == True:
                if rand_sev == 0:
                    brightness = None
                else:
                    brightness = iaa.imgcorruptlike.Brightness(
                        severity=rand_sev
                    )
            else:
                brightness = iaa.imgcorruptlike.Brightness(
                    severity=self.b_sev
                )
            if brightness is None:
                return img
            else:
                if pre_norm:
                    img = (img * 255).astype(np.uint8)
                    brightness_img = (brightness(image=img)) / 255
                else:
                    brightness_img = brightness(image=img)
                return brightness_img
        else:
            return img

    #     ----------

    def aug_elastic(self, img, pre_norm=False):

        """
        Applies elastic transformation to the image (set pre_norm
        to True if image has been normalized to [0,1]).
        """

        if self.e_sev is not None:
            rand_sev = np.random.randint(self.e_sev + 1)
            if self.randomize_params == True:
                if rand_sev == 0:
                    elastic = None
                else:
                    elastic = iaa.imgcorruptlike.ElasticTransform(
                        severity=rand_sev
                    )
            else:
                elastic = iaa.imgcorruptlike.ElasticTransform(
                    severity=self.e_sev
                )
            if elastic is None:
                return img
            else:
                if pre_norm:
                    img = (img * 255).astype(np.uint8)
                    elastic_img = (elastic(image=img)) / 255
                else:
                    elastic_img = elastic(image=img)
                return elastic_img
        else:
            return img

    #     ----------

    def aug_contrast(self, img, pre_norm=False):

        """
        Adusts contrast to a random subset of images (set pre_norm
        to True if image has been normalized to [0,1]).
        """

        if self.contrast == True:
            contrast = iaa.pillike.EnhanceContrast()
            if self.randomize_params == True:
                if np.random.randint(2) == True:
                    if pre_norm:
                        img = (img * 255).astype(np.uint8)
                        contrast_img = (contrast(image=img)) / 255
                    else:
                        contrast_img = contrast(image=img)
                else:
                    if pre_norm:
                        img = (img * 255).astype(np.uint8)
                        contrast_img = (contrast(image=img)) / 255
                    else:
                        contrast_img = img
            else:
                if pre_norm:
                    img = (img * 255).astype(np.uint8)
                    contrast_img = (contrast(image=img)) / 255
                else:
                    contrast_img = contrast(image=img)
            return contrast_img
        else:
            return img

    #     ----------

    def aug_sharpness(self, img, pre_norm=False):

        """
        Adjusts sharpness to a random subset of images (set pre_norm
        to True if image has been normalized to [0,1]).
        """

        if self.sharpness == True:
            sharpness = iaa.pillike.EnhanceSharpness()
            if self.randomize_params == True:
                if np.random.randint(2) == True:
                    if pre_norm:
                        img = (img * 255).astype(np.uint8)
                        sharpness_img = (sharpness(image=img)) / 255
                    else:
                        sharpness_img = sharpness(image=img)
                else:
                    if pre_norm:
                        img = (img * 255).astype(np.uint8)
                        sharpness_img = (sharpness(image=img)) / 255
                    else:
                        sharpness_img = img
            else:
                if pre_norm:
                    img = (img * 255).astype(np.uint8)
                    sharpness_img = (sharpness(image=img)) / 255
                else:
                    sharpness_img = sharpness(image=img)
            return sharpness_img
        else:
            return img

    #     ----------

    def simple_augment(self, img, order=None, pre_norm=False):

        """
        Performs the augmentations (for which a parameter was provided) in
        a particular order.

        Arguments:
            img (array): image to augment.

        Keyword Arguments:
            order (list) optional: list of indices (integer type) to
            determine the order in which the transformation functions are
            applied. Note that the transformation functions are ordered
            alphabetically by default. Defaults to None.
            -
            pre_norm (bool) optional: whether or not the input image
            pixel values have been normalized to the range [0,1]. If True,
            the pixel values will be de-normalized prior to augmentation.
            Defaults to False.

        Returns:
            Augmented image.
        """
        if pre_norm:
            img = (img * 255).astype(np.uint8)

        if order:
            aug_order = order
        else:
            aug_order = np.arange(self.argno)
        for i in aug_order:
            function = self.f_list[i]
            outp = function(img)
            img = outp
        if pre_norm:
            return outp / 255
        else:
            return outp

    #     ----------

    def random_augment(self, img, full_set=True, pre_norm=False):

        """
        Performs the augmentations (for which a parameter was provided) in
        a random order.

        Arguments:
            img (array): image to augment.

        Keyword Arguments:
            full_set (bool) optional: whether or not to apply all of the
            transformation functions for which a parameter was provided in
            the constructor upon initialization. If False, will apply a
            random subset of the transformations. Defaults to True.
            -
            pre_norm (bool) optional: whether or not the input image
            pixel values have been normalized to the range [0,1]. If True,
            the pixel values will be de-normalized prior to augmentation.
            Defaults to False.

        Returns:
            Augmented image.
        """
        if pre_norm:
            img = (img * 255).astype(np.uint8)
        if full_set:
            aug_order = np.random.choice(
                np.arange(self.argno), size=self.argno, replace=False
            )
        else:
            aug_order = np.random.choice(
                np.arange(self.argno),
                size=np.random.randint(self.argno),
                replace=False,
            )
        for i in aug_order:
            function = self.f_list[i]
            outp = function(img)
            img = outp
        if pre_norm:
            return outp / 255
        else:
            return outp

    #     ----------

    def display(
        self, img, n_rows, n_cols, augment_type=None, order=None
    ):

        """
        Applies the Augmenter to an input image according to the
        parameters given, in order to visualize augmentations.

        Arguments:
            img (array): image to augment.
            -
            n_rows (int): number of rows of augmented images to
            display.
            -
            n_cols (int): number of columns of augmented images to
            display.

        Keyword Arguments:
            augment_type (str) optional: either "random" or "simple".
            If "random", the class' random_augment method will be used
            for the display. If "simple", the simple_augment method
            will be used. If None, "random_augment" is used. Defaults to
            None.
            -
            order (list) optional: list of indices (integer type) to
            determine the order in which the transformation functions are
            applied. Note that the transformation functions are ordered
            alphabetically by default. Only relevant if using "simple"
            as the "augment_type" (see above). Defaults to None.

        Returns:
            Visualization of augmented image.
        """

        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = "Helvetica"
        plt.rcParams["text.color"] = "#333F4B"

        fig = plt.figure(figsize=(12, 9))
        ax1 = fig.add_subplot(n_rows, n_cols, 1)
        if img.shape[2] == 1:
            ax1.imshow(np.squeeze(img, 2))
        else:
            ax1.imshow(img)
        ax1.set_title("Original", fontweight="bold")
        ax1.set_xticks([])
        ax1.set_yticks([])
        for i in tqdm(range(2, (n_rows * n_cols) + 1)):
            if augment_type == "simple":
                aug_img = self.simple_augment(img, order=order)
            else:
                aug_img = self.random_augment(img)
            ax = fig.add_subplot(n_rows, n_cols, i)
            if aug_img.shape[2] == 1:
                ax.imshow(np.squeeze(aug_img, 2))
            else:
                ax.imshow(aug_img)
            ax.set_xticks([])
            ax.set_yticks([])

        fig.tight_layout()
        plt.show()

    #     ----------

    def display_sample(
        self,
        source_type,
        source,
        n_rows,
        n_cols,
        augment_type=None,
        order=None,
        pre_norm=False,
    ):

        """
        Fetches a sample image from a collection of images located in
        a main directory or an uptools.Image_Dataset and applies the
        Augmenter according to the parameters given, in order to
        visualize augmentations prior to running on multiple images.

        Arguments:
            source_type (str): type of the source from which the sample
            is drawn. Can either be "path" or "ids". If "path", the
            source argument will be taken as the path to the directory
            containing the images. If "ids", the source argument will
            be taken as the name of an (initialized) imgo.utpools
            Image_Dataset object.
            -
            source (str or Image_Dataset object): source of the images
            from which the sample is drawn. Can either be the base path
            to the directory containing images (or class subdirectories)
            if the source_type argument is given as "path", or an
            (initialized) imgo.uptools Image_Dataset object if the
            source_type argument is given as "ids".
            -
            n_rows (int): number of rows of augmented images to
            display.
            -
            n_cols (int): number of columns of augmented images to
            display.

        Keyword Arguments:
            augment_type (str) optional: either "random" or "simple".
            If "random", the class' random_augment method will be used
            for the display. If "simple", the simple_augment method
            will be used. If None, "random_augment" is used. Defaults to
            None.
            -
            order (list) optional: list of indices (integer type) to
            determine the order in which the transformation functions are
            applied. Note that the transformation functions are ordered
            alphabetically by default. Only relevant if using "simple"
            as the "augment_type" (see above). Defaults to None.
            -
            pre_norm (bool) optional: whether or not the input image
            pixel values have been normalized to the range [0,1]. If True,
            the pixel values will be de-normalized prior to augmentation.
            Only relevant if using "path" mode. Defaults to False.

        Returns:
            Visualization of augmented image.
        """

        if source_type == "path":

            from imgo.uptools import img_to_df

            df = img_to_df(source)
            sample = df.sample()
            img = imageio.imread(sample.iloc[0]["image"])

        elif source_type == "ids":
            if source.shadow["data"][0] is None:
                img = source.shadow["train"][0][
                    np.random.randint(
                        source.shadow["train"][0].shape[0]
                    )
                ]
            else:
                img = source.shadow["data"][0][
                    np.random.randint(source.shadow["data"][0].shape[0])
                ]
        else:
            raise Exception(
                "Must choose valid source type: either 'path' or 'ids'."
            )

        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = "Helvetica"
        plt.rcParams["text.color"] = "#333F4B"

        fig = plt.figure(figsize=(12, 9))
        ax1 = fig.add_subplot(n_rows, n_cols, 1)
        if img.shape[2] == 1:
            ax1.imshow(np.squeeze(img, 2))
        else:
            ax1.imshow(img)
        ax1.set_title("Original", fontweight="bold")
        ax1.set_xticks([])
        ax1.set_yticks([])
        for i in tqdm(range(2, (n_rows * n_cols) + 1)):
            if augment_type == "simple":
                aug_img = self.simple_augment(img, order=order)
            else:
                aug_img = self.random_augment(img)
            ax = fig.add_subplot(n_rows, n_cols, i)
            if aug_img.shape[2] == 1:
                ax.imshow(np.squeeze(aug_img, 2))
            else:
                ax.imshow(aug_img)
            ax.set_xticks([])
            ax.set_yticks([])

        fig.tight_layout()
        plt.show()

    #     ----------

    def augment_flow(
        self,
        base_path,
        augment_type=None,
        order=None,
        class_selection=None,
        number=None,
        size=None,
        save=False,
    ):

        """
        Fetches all the images from a collection of images located in a
        main directory and applies a series of random augmentations to
        them according to the parameters given to an (initialized)
        Augmenter object. Depending on the parameters given, a subset of
        the images can be augmented, or the entire dataset can be
        augmented.

        Note that this function requires the image data to be in image
        form (as opposed to numpy-arrays) and located in the relevant
        directories, ie a single image directory (the base path) or a
        subdirectory for each class within the image directory.

        Note also that running the function more than once on the same
        base path will overwrite previously augmented images (unless
        their names have been changed).

        Arguments:
            base_path (str): path to the directory containing images or
            class subdirectories.

        Keyword Arguments:
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
            class_selection (list) optional: list of class names on which
            to perform the augmentation. If not given, will apply to all
            the classes in the dataset. Defaults to None.
            -
            number (int) optional: number of images to augment (whether
            in total or per class will be inferred by the directory
            structure of the base_path argument provided). Note that if
            neither number nor size arguments are given, each image from
            each class will be augmented once, thereby doubling the size
            of the dataset. Defaults to None.
            -
            size (int) optional: desired size of the dataset resulting
            after the augmentation (whether in total or per class will be
            inferred by the directory structure of the base_path argument
            provided); provided as an alternative to the number argument.
            Note that if neither number nor size arguments are given,
            each image from each class will be augmented once, thereby
            doubling the size of the dataset. Defaults to None.
            -
            save (bool) optional: whether or not to save the images in
            the directory given as the base_path argument. If the
            directory structure has no classes, these will be saved in a
            newly created directory: 'imgo_output/augtools/augment_flow'.
            If the directory structure has subdirectories for each class,
            the augmented images will be saved in the corresponding class
            subdirectory. Defaults to False.

        Returns:
            aug_imgs (numpy-array): augmented images in numpy-array form
            (if save is False).
        """

        from imgo.uptools import img_to_df
        from imgo.uptools import read_img_df

        all_class_arrays = []
        all_aug_indices = []
        aug_imgs = []
        df = img_to_df(base_path)
        no_class = False

        if class_selection:
            if type(class_selection) is not list:
                raise Exception(
                    f"Class selection must be a list; {type(class_selection)} given."
                )
            else:
                df = df.loc[df["class"].isin(class_selection)]

        if len(df.loc[df["class"] == "no_class"]) != 0:
            if int(
                df.loc[df["class"] == "no_class"][
                    "class"
                ].value_counts()
            ) == len(df):
                no_class = True

        class_list = list(df["class"].unique())
        for c in class_list:
            class_arrays = read_img_df(df, class_name=c)
            all_class_arrays.append(class_arrays)

        if (number is not None) and (size is not None):
            raise Exception(
                "Cannot accept both number and size arguments."
            )

        elif (number is not None) and (size is None):
            for a in all_class_arrays:
                to_augment = np.random.randint(a.shape[0], size=number)
                all_aug_indices.append(to_augment)

        elif (number is None) and (size is not None):
            for a in all_class_arrays:
                if size - a.shape[0] < 0:
                    raise Exception(
                        "Cannot accept size argument smaller than number of images."
                    )
                else:
                    to_augment = np.random.randint(
                        a.shape[0], size=size - a.shape[0]
                    )
                    all_aug_indices.append(to_augment)

        else:
            for a in all_class_arrays:
                to_augment = np.array(range(0, a.shape[0]))
                all_aug_indices.append(to_augment)

        if save:
            if no_class == True:
                my_path = "imgo_output/augtools/augment_flow"

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

                for i, j in tqdm(
                    enumerate(all_aug_indices[0]),
                    total=len(all_aug_indices[0]),
                ):
                    if augment_type == "simple":
                        aug_img = self.simple_augment(
                            all_class_arrays[0][j], order=order
                        )
                    else:
                        aug_img = self.random_augment(
                            all_class_arrays[0][j]
                        )
                    imageio.imwrite(f"{r}/aug_{i+1}.jpg", aug_img)
                    aug_imgs.append(aug_img)
                print("Augmented images saved successfully.")

            else:
                for x in range(len(all_class_arrays)):
                    for i, j in tqdm(
                        enumerate(all_aug_indices[x]),
                        total=len(all_aug_indices[x]),
                    ):
                        if augment_type == "simple":
                            aug_img = self.simple_augment(
                                all_class_arrays[x][j], order=order
                            )
                        else:
                            aug_img = self.random_augment(
                                all_class_arrays[x][j]
                            )
                        imageio.imwrite(
                            f"{base_path}/{class_list[x]}/{class_list[x]}_aug_{i+1}.jpg",
                            aug_img,
                        )
                        aug_imgs.append(aug_img)
                print(
                    "Augmented images saved successfully in class subdirectories."
                )
        else:
            if no_class == True:
                for i, j in tqdm(
                    enumerate(all_aug_indices[0]),
                    total=len(all_aug_indices[0]),
                ):
                    if augment_type == "simple":
                        aug_img = self.simple_augment(
                            all_class_arrays[0][j], order=order
                        )
                    else:
                        aug_img = self.random_augment(
                            all_class_arrays[0][j]
                        )
                    aug_imgs.append(aug_img)

            else:
                for x in range(len(all_class_arrays)):
                    for i, j in tqdm(
                        enumerate(all_aug_indices[x]),
                        total=len(all_aug_indices[x]),
                    ):
                        if augment_type == "simple":
                            aug_img = self.simple_augment(
                                all_class_arrays[x][j], order=order
                            )
                        else:
                            aug_img = self.random_augment(
                                all_class_arrays[x][j]
                            )
                        aug_imgs.append(aug_img)

            return np.array(aug_imgs)
