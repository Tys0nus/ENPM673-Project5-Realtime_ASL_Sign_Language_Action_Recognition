ENPM 673 - Project 5
=======================================

Realtime ASL Sign Language Action Recognition- Empowering Communication and Inclusion
=======================================

The goal of this project is to design and implement a real-time sign language detection system that can accurately recognize and interpret American Sign  Language (ASL) gestures and display the corresponding text

Requirements
------------

The following libraries and dependencies are required to run the project:

-   Python 3.x
-   NumPy
-   OpenCV
-   Mediapipe
-   Tensorflow
-   sklearn
-   Matplotlib
-   os

Project Structure
-----------------

The project contains the following files:

-   `python main.py`: Main script to run the project using custom trained weights.
-   `python data_collection.py`: Script for data collection pipeline. [dependencies - actionset.py : note all the signs for collection]
-   `python model.py`: Script to train the model for collected data.
-   `python pre_process.py`: Data Split into train-test data.
-   `python actionset.py`: Script involving data labels.

-   `DataProcess`: Folder containing all the CUSTOM COLLECTED DATA using webcam.

Usage
-----

CUSTOM DATA SET COLLECTION AND TRAINING

-   To collect custom data, update the labels in `python actionset.py`.
-   Run `python data_collection.py` to start the data collection
-   To train a custom model, run 'python model.py' on the custom data which will generate a model_weights.h5 file


-----
RUN THE MODEL

To run the project, simply run the `main.py` script to run using trained weights - `model_weights.h5`:


`python main.py`


Successful demonstration of Realtime ASL Gesture Recognition as probabilities


GITHUB Repository
-----

https://github.com/Tys0nus/ENPM673-Project5-Realtime_ASL_Sign_Language_Action_Recognition.git

REALTIME DEMONSTRATION USING WEBCAM 
-----

Link - https://drive.google.com/file/d/1TZdnJFsovd6PuLFOZx0nfASWf6ZMv9ro/view?usp=sharing


Author
-----
Aditya Chaugule
Graduate Student
Master of Engineering in Robotics
University of Maryland, College Park
aditya97@umd.edu


Ankur Chavan
Graduate Student
Master of Engineering in Robotics
University of Maryland, College Park
achavan1@umd.edu
