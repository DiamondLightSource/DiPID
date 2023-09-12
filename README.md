## DiPID - Diffraction Peak Identifier

This repository contains two separate machine learning models designed to identify and locate diffraction peaks in scans taken on the I16 beamline at Diamond Light Source. Each produces outputs in the form of a binary mask to segment the image into 'peak' and 'background' regions.

The models can be used on individual images or on entire scans, and the data from full scans can be visualised in 3D to help overcome any missed peaks in individual detections.

# Transfer Learning Model (TLM)

This model is built on a ResNet50 backbone, with a new MaskRCNN and FastRCNN head attached on top. During training only the head weights are updated, preserving the pre-trained weights of the backbone. 

When predicting, the model generates a list of binary masks, each corresponding to a peak. These are thresholded by a confidence level, and can also be filtered using non-max suppression to remove overlapping masks.

The available weights were trained for 475 epochs on a labelled dataset of 9 commissioning scans on the I16 beamline, ending with an average IoU of 0.46 for the test set (using a confidence threshold of 0.4).

# Self-Supervised Learning Model (SSLM)

This model is a simple CNN (default of 3 layers and 100 channels). It uses a weighted combination of 3 loss functions to train on an unlabelled dataset.

When predicting, the model generates a tensor where the value of each pixel is the most likely channel / cluster for that pixel to belong to. This can then be filtered into 'peak' and 'background' if it can be identified which channel is used for which features in the image.

The available weights were trained for 40 epochs on an unlabelled dataset of 9 commissioning scans on the I16 beamline, ending with an average IoU of 0.12 for the test dataset. The model must be initialised in the default configuration for these weights to be compatible. This pre-trained model uses channel 66 for central parts of a peak, and channel 13 for the blurry region around a peak.

# Joint Predictor

The easiest way to make predictions with the available weights is to load the Joint Predictor (JP) class. This has four main functions available:
- `predict_one_image()`: takes a folder, nexus file and image number within the file and generates masks with both models. Use the `use_nms` argument to switch on and off the nms filtering. Set the `use_ssl` or `use_tlm` arguments to false to not use a specified model in the prediction. The masks are returned in the format `ssl_masks, 'tlm_masks`.
- `predict_scan()`: takes a folder and nexus file and generates predictions for every image in the scan. Use the `start_val` and `end_val` arguments to only predict for a subset of the images. Use `use_ssl` and `use_tlm` as in the `predict_one_image` function. Outputs are in the form of `ssl_masks, tlm_masks`, where each is a dictionary with the key as the image number and the value the single image prediction. 
- `generate_3d_plot()`: takes a dictionary of predictions and compiles them into a 3D volume. The `scan_type` variable must be set so that the masks can be handled appropriately, either to `'tlm'` or `'ssl'`.
- `show_image_predictions()`: displays the masks predicted by the model overlaid on original images. Also shows the outputs with log scaling. Takes the same inputs as the `predict_one_image` function.