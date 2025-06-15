# localisation_INNs
Problem statement: localisation of a robot in known environment with camera input using Invertible Neural Networks 

This project makes use of code from the original repository: https://github.com/zzangupenn/Local_INN/ and the paper [Link](https://arxiv.org/abs/2209.11925).

Folder structure:

1. The folder "local_inn" is a ros1 package. It contains 2 types of scripts. one script (starting with "learned_nn_...py") load the trained model and the other scripts( starting with "inference...py) communicates with the trained model using ros services and plots the inferred pose and groudn truth pose. The ground truth test path is supplied in a csv file in inference scripts. You can also modify it to send ground truth poses and captured images in real time.


2. The "training_scripts" folder has scripts for training the INN model. The script names containing the world "sparse" are used to training seprate model for each parameter in the 3D pose, where as the other script learns all the 3 DOF pose using a single model

3. gazebo_plugin_img_collection: this folder contains gezebo plugin for collecting training images. this method is much faster than collecting images from volta robot using ros topics


SETUP:

1. Install ROS1 and gazebo on ubuntu 20.04.
2. Copy volta packages as mentioned in the github repo: https://github.com/airl-iisc/CPS280/tree/main/final_assignment
3. Copy the folder "local_inn" package and build the code.
4. Run the ros nodes to estimate the pose from trained model and plot the graphs



Training images data: https://drive.google.com/file/d/1_PPWGzl2Q0T1tcOdbzc0yhiI1xP4IBwl/view?usp=sharing

