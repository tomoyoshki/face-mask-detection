# CS 445 FA 21 Final Project

rd10, tkimura4, yaowenc2

## Face Mask detection

This is the final project for Computational Photography offered by University of Illinois at Urbana Champaign

In the era of COVID-19, although with vaccines, the primary line of defense still lies upon face masks. In many countries, we have seen that there are manual people checking whether or not people are wearing masks properly. Therefore, the idea of creating a mask detector sparks our interest. In this project, we aim to apply computational photography, computer vision, and machine learning to create a face mask detector. During this process, we hope to apply what we have learned during the class and challenge ourselves with essential techniques in Computer Vision that can potentially improve our world.

### Demonstration

#### Two Classes Detection

<img src="./assets/two_classes.gif" />

#### Three Classes Detection

<img src= "./assets/three_classes.gif" />

### Technologies

- Python 3.7
- Tensorflow 2 / Keras
- Pytorch
- Opencv

### Directories

```
Project
├── saved_model
│	├── cnn_model 				# saved model used for CNN model
│		├── keras_metadata.pd   
│		└── saved_model.pb
│   	├── resnet_model			# saved model used for ResNet model
│		└── model.pth
├── assets					# assets
│	├── two_classes.gif
	└── three_classes.gif
├── camera.py 					# CNN network
├── camera2.py 					# resNet
├── CNNfromScratch.ipynb			# CNN from scratch training and predicting
├── haarcascade_frontalface_default.xml		# face detection harrcascade source file
└── mask_detection_3.ipynb 			# ResNet training and predicting

```

### Data

#### Data collection

Our training and validation data is based on this [Dataset](https://www.kaggle.com/vijaykumar1799/face-mask-detection) we found on Kaggle.

There are more than 8000 images in this dataset, which includes various images from data augmentation.

#### Data preprocessing

We split our dataset into three directories with the labels as the directory names. For the two classes classification, we are only using the `with_mask` and `without_mask`, and for the three classes, we are using all. 

```
Dataset
├── mask_weared_incorrect
├── with_mask
├── without_mask
```

### Face detection

We used the `opencv` built-in `CascadeClassifier` with Haar Cascade data source to detect the faces in a frame.

### Project flow

#### CNN from Scratch + CNN Classifier

We implemented the convolution and max pooling part of the CNN from scratch. We recommend you to look at it and
understand the .
We then utilized Tensorflow framework for our two classes mask classifier. We have uploaded the model file correlated
with the GIF under the `saved_model/CNN_model` directory. You could directly run the `camera.py` file.

#### ResNet 34

We also applied ResNet 34 model for the three classes mask classifier. We expect the user to first run the CNN model, and then try this ResNet 34 model. We have uploaded the model file correlated with the GIF under the `saved_model/resnet_model` directory. You could directly run the `camera2.py` file. However, the `model.load()` function is currently losing information. Therefore, to achieve the best result, we are also offering the Jupyter notebook (`mask_detection_3.ipynb `) we used. This notebook provides GPU support, so you can easily train them and produce the desirable outcome.

## Contact

Feel free to contact us with any concern

- Tony Chang (tonychang04): yaowenc2@illinois.edu
- Charles Dai (CharlesDaiii): rd10@illinois.edu
- Tommy Kimura (tomoyoshki): tkimura4@illinois.edu
