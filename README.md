# IMGO

## Process, Augment, and Balance Image Data

This library is designed to facilitate the preprocessing phase of image classification projects in order to get into the fun part: training the models!

### Features:

Imgo is composed of two modules: **uptools** and **augtools**.

**Uptools** helps to streamline various image data preprocessing tasks, such as:

 - Reading images from a local disk
 - Rescaling images
 - Normalizing and standardizing pixel values
 - Converting image datasets into numpy-arrays
 - One-hot-encoding label data
 - Splitting image datasets into training, validation, and testing subsets
 - Merging data subsets into a single dataset
 - Saving numpy-arrays as images in class subdirectories
 
![imgo_up_demo](aux/imgo_up_demo.jpg)
 
**Augtools** allows the user to quickly and efficiently apply augmentation to image data. With Augtools, users can perform the following augmentation tasks using very few lines of code:

 - Apply a powerful collection of transformation and corruption functions
 - Augment images saved on a local disk
 - Save augmented images in class subdirectories
 - Augment entire image datasets
 - Augment training data in place in preparation for machine learning projects
 - Rebalance class sizes by generating new training images

![imgo_aug_demo](aux/imgo_aug_demo.jpg)

### Installation

It's as easy as `pip install imgo`!

### Quickstart

Have a look at the demos [here](https://github.com/elbydata/imgo/tree/master/demos)!

### Documentation

Documentation is currently available in the form of docstrings.

### Requirements and Dependencies

Please see the `requirements.txt` file for all requirements and dependencies.
 
### Support

The source code is available [here](https://github.com/elbydata/imgo/tree/master/imgo).
Please direct any queries or issues to info@elbydata.com.

### Issues / To do

Some functions currently employ ragged arrays which are deprecated in the latest versions of NumPy. This affects all functions that work with non-standard or inconsistent image dimensions.

### License

The project is licensed under the MIT license.

### Acknowledgements

Some of the **augtools** library is built as a wrapper around **Imgaug**, a powerful image augmentation library. For more information, please see https://imgaug.readthedocs.io/en/latest/.