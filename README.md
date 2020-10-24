# IMGO

## Compile, Process, and Augment Image Data

This library is designed to facilitate the preprocessing phase of image classification projects. 

### Features:

Imgo is composed of two distinct but related modules: **Uptools** and **Augtools**.

**Uptools** helps to streamline various image data preprocessing tasks, such as:

 - Reading images from a local disk
 - Rescaling images
 - Normalizing and standardizing pixel values
 - Converting image datasets into numpy-arrays
 - One-hot-encoding label data
 - Splitting image datasets into training, validation, and testing subsets
 - Saving numpy-arrays as images in class subdirectories
 
**Augtools** allows the user to quickly and efficiently apply augmentation to image data. With Augtools, users can perform the following augmentation tasks using very few lines of code:

 - Apply a small but powerful collection of transformation and corruption functions
 - Augment images saved on a local disk
 - Save augmented images in class subdirectories
 - Augment entire image datasets
 - Augment training data in place in preparation for machine learning projects
 
Here is a sample image augmented using Imgo:

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

### License

The project is licensed under the MIT license.

### Acknowledgements

The **Augtools** library is built on top of **Imgaug**, a powerful image augmentation library. For more information, please see https://imgaug.readthedocs.io/en/latest/.