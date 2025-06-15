# localisation_INNs
Problem statement: localisation of a robot in known environment with camera input using Invertible Neural Networks 

This project makes use of code from the original repository: https://github.com/zzangupenn/Local_INN/ and the paper [Link](https://arxiv.org/abs/2209.11925).

Folder structure:

First install ROS1 and then add volta packages.

The folder "local_inn" is a ros1 package. It contains 2 types of scripts. one script (starting with "learned_nn_...py") load the trained model and the other scripts( starting with "inference...py) communicates with the trained model using ros services and plots the inferred pose and groudn truth pose. The ground truth test path is supplied in a csv file in inference scripts. 


The "training_scripts" folder has scripts for training the INN model. The script names containing the world "sparse" are used to training seprate model for each parameter in the 3D pose, where as the other script learns all the 3 DOF pose using a single model

