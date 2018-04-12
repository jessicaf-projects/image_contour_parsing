# Dicom file parser for image segmentation

This code prepares a set of dicom images and manually drawn contours for input into a convolutional neural network for image segmentation.

## Parsing and matching the contours

In order to verify that the contours are parsed correctly, I plotted the images and contours alongside each other to verify that these were correct. Some examples can be found in the folder testing/test_image_matching.

I made minor changes to integrate it into an existing codebase. These included changing the inputs and outputs to the functions for parsing and reading the contours.

## Building the model training pipeline

The pipeline for parsing and matching the contours worked well with the function for generating the epochs. However, this could be better streamlined in the future by incorporating it into the epoch object rather than being a standalone function.

In order to verify that the pipeline was working correctly, I generated some unit tests (test_epochs.py) and generated folders containing the inputs and targets for multiple epochs and batches (generate_and_test_outputs.py). Running test_epochs.py will test that the randomization and generation of the epochs and batches are working properly.

As it stands, the unit tests are not comprehensive, and future work includes generating synthetic images and contours to verify that the randomization and mask generation provide the expected pixel numbers and values.

To improve this pipeline, I can add better error-checking to account for many more possible cases of missing dicom files, and provide better control over the randomization/generation/exporting of the batch and epoch information. I can also integrate PyTorch or TensorFlow functions to train a convolutional neural network. Finally, I can modify this project so that the functions are command-line executable.




