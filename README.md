# Cat Classification

## Overview

This git repository contains code and configurations for implementing a Convolutional Neural Network to classify images containing cats or dogs. The data was sourced from the [dogs-vs-cats](https://www.kaggle.com/competitions/dogs-vs-cats/overview) Kaggle competition, and also from [freeimages.com](https://www.freeimages.com/) using a web scraper.

Two models were trained to classify the images; an AlexNet8 model via Keras and a VGG16 model via Torch.

Docker containers were used to deploy the application on an EC2 spot instances in order to scale up hardware and computation power. 

![Workflow](doc/catclassifier.jpg)

## Analysis Results

Both models were trained using a variety of image transformations, early stopping, stochastic gradient descent, learning rate reduction and cross entropy loss criterion.

See the analysis results notebook for a summary of the project; including image processing, CNN architecture and model performance.
* https://nbviewer.org/github/oislen/CatClassifier/blob/main/report/torch_analysis_results.ipynb

## Running the Application (Windows)

### Anaconda

### Docker

The application docker container is available on dockerhub here:

https://hub.docker.com/repository/docker/oislen/cat-classifier
