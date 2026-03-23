# SORTWISE - WASTE SEGMENTATION
## Real-Time Classifier
OVERVIEW
--------
Sortwise is a real-time waste classification system that uses a webcam feed and
a trained machine learning model to identify and categorize waste items into one
of eight categories. It is designed to assist with smart waste sorting by giving
instant visual feedback on what type of waste is in view of the camera.

Supported waste categories:
  - Paper
  - Plastic
  - Can
  - Cardboard
  - Cloth
  - Empty (no item / empty container)
  - Humans
  - Glass


HOW IT WORKS
------------
The program captures live video from a connected webcam. Each frame is resized
to 224x224 pixels and normalized before being passed to a MobileNetV2-based
neural network (trained via Google Teachable Machine). The model outputs a
confidence score for each category, and the highest-scoring label is displayed
on screen in real time.

Press ESC at any time to exit the program.


FILES
-----
WasteSeg.py          - Main application script (webcam + inference loop)

fix_model.py          - One-time script to patch the model file for
                          compatibility with newer versions of Keras/TensorFlow
                          
keras_model.h5        - Original trained model (exported from Teachable Machine)

keras_model_fixed.h5  - Patched model file produced by fix_model.py
                          (use this one at runtime)
                          
labels.txt            - List of class labels corresponding to model outputs

WasteSeg_original.py   - Original main application script

REQUIREMENTS
------------
  Python 3.13+
  tensorflow >= 2.20.0
  tf_keras
  opencv-python
  h5py
  numpy

Install dependencies with:

  pip install tensorflow tf_keras opencv-python h5py numpy


SETUP AND USAGE
---------------
1. Run the model fix script once (only needed the first time):

      python fix_model.py

   This will produce keras_model_fixed.h5 in the same directory.

2. Run the main application:

      python Waste_Seg.py

3. Point your webcam at a waste item. The detected category and confidence
   score will be shown as an overlay on the live video window.

4. Press ESC to quit.


--------------------------------------------------------------------------------
UPDATE NOTES -- WHY THIS PROJECT WAS MODIFIED
--------------------------------------------------------------------------------

The original version of this project used cvzone's Classifier module to load
the Keras model directly. This worked fine with older versions of TensorFlow
and Keras (2.x series), which is the environment Google Teachable Machine uses
to export models.

The project was updated because of a breaking incompatibility introduced in
Keras 3.x (which ships with TensorFlow 2.16 and later). Python 3.13 cannot
use TensorFlow versions older than 2.20, and TensorFlow 2.20 bundles Keras 3.x
by default. This created two specific problems:

  PROBLEM 1 -- Unrecognized argument 'groups':
  The model config saved by Teachable Machine includes a 'groups': 1 parameter
  in DepthwiseConv2D and Conv2D layer definitions. Keras 2.x accepted this
  argument silently, but Keras 3.x raises a hard ValueError and refuses to
  load the model.

  PROBLEM 2 -- Sequential model deserialization failure:
  Even after patching the 'groups' argument, Keras 3.x uses an entirely
  different serialization format internally. It could not reconstruct the
  nested Sequential/Functional model architecture that Teachable Machine
  produces, resulting in a tensor shape mismatch error.

  ATTEMPTED FIX -- TensorFlow downgrade:
  Downgrading to TensorFlow 2.12 (which uses Keras 2.x and has no issues
  loading Teachable Machine models) was not possible because that version
  does not support Python 3.13. The earliest TensorFlow version available
  for Python 3.13 is 2.20.

THE SOLUTION
  The fix works in two parts:

  1. fix_model.py surgically rewrites the model's stored configuration inside
     the .h5 file, removing all 'groups' keys from every layer definition.
     It preserves all trained weights exactly as they are.

  2. The main script loads the patched model using tf_keras, which is the
     standalone Keras 2 compatibility package that TensorFlow 2.16+ ships
     separately. This allows the old model architecture to be deserialized
     correctly without requiring a TensorFlow downgrade.

  The model weights and trained behavior are completely unchanged. Only the
  metadata describing the model architecture was patched.

--------------------------------------------------------------------------------

NOTES
-----
- The model was originally trained using Google Teachable Machine, which
  exports in the Keras 2 .h5 format (MobileNetV2 backbone).
- keras_model.h5 (the original file) is kept for reference but should NOT
  be used directly at runtime on Python 3.13 + TensorFlow 2.20+.
- Always use keras_model_fixed.h5 as the model path in Waste_Seg.py.

================================================================================
