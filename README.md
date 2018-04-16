# Dicom file parser for image segmentation

This code prepares a set of dicom images and manually drawn contours for input into a convolutional neural network for image segmentation.

## Example usage

```
from epochs import create_epoch

epoch = create_epoch('final_data')
images, targets = epoch.get_current_batch()
while images is not None:
    images, targets = epoch.get_current_batch()
```

The above will iterate through each batch in the epoch. In order to create a new epoch, use:

```
epoch.new_epoch()
```

## Parsing and matching the contours

In order to verify that the contours are parsed correctly, I plotted the images and contours alongside each other. Some examples can be found in the folder testing/test_image_matching. This folder contains some examples of matched contours, dicom files, and masks.

I made minor changes to integrate it into an existing codebase. These included changing the inputs and outputs to the functions for parsing and reading the contours.

## Building the model training pipeline

The pipeline for parsing and matching the contours worked well with the function for generating the epochs. However, this could be better streamlined in the future by incorporating it into the epoch object rather than being a standalone function.

In order to verify that the pipeline was working correctly, I generated some unit tests (test_epochs.py) and generated folders containing the inputs and targets for multiple epochs and batches (generate_and_test_outputs.py). Running test_epochs.py will test that the randomization and generation of the epochs and batches are working properly.

As it stands, the unit tests are not comprehensive, and future work includes generating synthetic images and contours to verify that the randomization and mask generation provide the expected pixel numbers and values.

To improve this pipeline, I can add better error-checking to account for many more possible cases of missing dicom files, and provide better control over the randomization/generation/exporting of the batch and epoch information. I can also integrate PyTorch or TensorFlow functions to train a convolutional neural network. Finally, I can modify this project so that the functions are command-line executable.

## Changes to the code to include the o-contour

In order to modify the code to parse the o-contour, I used the o-contour filenames to search for the corresponding Dicom files and i-contour files. The reason for this was that not all of the Dicom and i-contour files had an o-contour file. In the event that not all o-contour files had an i-contour or Dicom file, the try/except statement would take care of this.

I also noted when plotting the i-contours and o-contours atop the images, that the last files listed in the link file did not appear to have accurate o-contour files. Because of this, I created a new link file that omitted the last file.

To test that the i-contours and o-contours were correctly placed, I generated plots that are located in testing/test_image_matching showing the images, contours, and masks.

## Analysis for Heuristic LV segmentation

The analysis for LV segmentation is located in analysis/Heuristic_LV_Segmentation_approaches.ipynb. The histogram in that file shows that a simple thresholding scheme (thresholding all of the images based on a specific pixel value in order to get the i-contour) would not be sufficient. There is considerable overlap in the average pixel intensities between the inner blood pool and outer contour, so selecting a specific value for thresholding likely wouldn't work. There is also considerable overlap between the individual pixel intensities, as shown in the plot above.

Other heuristic methods have the potential to work in this case. Specifically, methods that don't rely on the absolute pixel value but instead rely on the relative pixel value may be useful. One such method would start at the o-contour boundary and examine pixel values traveling toward the centroid of the o-contour, looking for a sudden increase in the pixel values. The point at which the values increase suddenly would define the i-contour boundary.

Using the intensities directly may work if we take the average over multiple neighboring pixels (but not all) in case there are a small number of pixels with very high or low values. Finally, thresholding in combination with image dilation/erosion may work for using the intensities to determine the inner contour.

